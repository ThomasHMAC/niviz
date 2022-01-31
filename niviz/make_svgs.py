from __future__ import annotations

import os
import argparse
import yaml
from multiprocessing import Pool
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def _get_package_name(config):
    '''
    Get package name from YAML specification or report file
    '''

    with open(config, 'r') as f:
        package_name = yaml.load(f, Loader=Loader)['package']
    return package_name


def _mksvg(interface):
    interface.run()


def _parse_var(s):
    key, value = [i.strip() for i in s.split("=")]
    return (key, value)


def _parse_vars(items):
    if items:
        return dict(_parse_var(item) for item in items)
    return {}


def single_image_util(args):
    '''
    Single image utility
    '''

    from niviz.node_factory import get_interface, ArgInputSpec

    out_path = args.out_svg
    specs = _parse_vars(args.set)
    spec = ArgInputSpec(name="single_image",
                        method=args.method,
                        bids_entities={},
                        interface_args=specs,
                        out_spec=out_path)
    interface = get_interface(spec=spec,
                              out_path=os.path.dirname(out_path),
                              make_dirs=False)
    interface.run()


def svg_util(args):
    '''
    SVG sub-command
    '''

    import niviz.node_factory
    import niviz.config

    arg_specs = niviz.config.fetch_data(args.spec_file, args.base_path)
    out_path = os.path.join(args.out_path, _get_package_name(args.spec_file))

    if not args.rewrite:
        arg_specs = [
            a for a in arg_specs
            if not os.path.exists(os.path.join(out_path, a._out_spec))
        ]

    interfaces = []
    for a in arg_specs:
        node = niviz.node_factory.get_interface(a, out_path)
        if node is not None:
            interfaces.append(node)

    if args.nthreads == 1:
        [_mksvg(i) for i in interfaces]

    with Pool(processes=args.nthreads) as pool:
        pool.map(_mksvg, interfaces)

    return


def report_util(args):
    '''
    Report generation utility
    '''

    from niworkflows.reports.core import run_reports

    package_path = os.path.join(args.base_path, _get_package_name(args.config))

    subject_list = args.subjects
    if args.subjects:
        subject_list = args.subjects
    else:
        subject_list = [
            s for s in os.listdir(package_path)
            if ('sub-' in s) and ('.html' not in s)
        ]

    [
        run_reports(args.output_dir,
                    s,
                    'NULL',
                    config=args.config,
                    reportlets_dir=args.base_path) for s in subject_list
    ]

    return


def cli():
    '''
    CLI Entry function
    '''

    p = argparse.ArgumentParser(
        description='Command line interface to Niviz to generate '
        'QC images from pipeline outputs')

    sub_parsers = p.add_subparsers(help='Niviz command modes')

    parser_svg = sub_parsers.add_parser('svg', help='SVG Generation utility')
    parser_svg.add_argument('base_path',
                            type=str,
                            help='Base path to pipeline outputs')
    parser_svg.add_argument('spec_file',
                            type=str,
                            help='Specification configuration file')
    parser_svg.add_argument('out_path',
                            type=str,
                            help='Base output path to create SVGs')
    parser_svg.add_argument('--nthreads',
                            type=int,
                            nargs="?",
                            const=1,
                            help="Number of threads to parallelize across")
    parser_svg.add_argument("--rewrite",
                            help="Overwrite existing SVG files",
                            action="store_true")
    parser_svg.set_defaults(func=svg_util)

    parser_report = sub_parsers.add_parser('report',
                                           help='Report Generation utility')
    parser_report.add_argument('base_path', type=str, help='Base path to SVGs')
    parser_report.add_argument('config',
                               type=str,
                               help='Path to report configuration')
    parser_report.add_argument('output_dir',
                               type=str,
                               help='Path to output reports')
    parser_report.add_argument('--subjects',
                               nargs='+',
                               help='List of subjects to generate reports for')

    parser_single = sub_parsers.add_parser('single',
                                           help='Generate single QC image')

    parser_single.set_defaults(func=single_image_util)
    parser_single.add_argument('method',
                               type=str,
                               help='Method to use to generate QC image')
    parser_single.add_argument('out_svg',
                               type=str,
                               help="Path to output SVG image")
    parser_single.add_argument('--set',
                               metavar='KEY=VALUE',
                               nargs='+',
                               help='Set a number of key-value pairs '
                               'to method arguments.')

    p.set_defaults(func=report_util)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    cli()
