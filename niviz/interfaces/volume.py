# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations
from typing import TYPE_CHECKING

from svgutils.transform import fromstring
from traits.trait_types import BaseInt
import nilearn.image as nimg
import nilearn.plotting as nplot
import nibabel as nib
from nipype.interfaces.base import File, traits, InputMultiPath
from nipype.interfaces.mixins import reporting
from niworkflows.viz.utils import (cuts_from_bbox, compose_view, extract_svg,
                                   robust_set_limits)
import niworkflows.interfaces.report_base as nrc

from ..node_factory import register_interface
"""
ReportCapable concrete classes for generating reports as side-effects
"""

if TYPE_CHECKING:
    from nipype.interfaces.base.support import Bunch
    from nibabel.nifti1 import Nifti1Image

# TODO: Create Identity base class


# Basic set of visualizations
class _IAnatInputSpecRPT(nrc._SVGReportCapableInputSpec):
    nii = File(exists=True,
               usedefault=False,
               resolve=True,
               desc="Anatomical Image to view",
               mandatory=True)

    n_cuts = BaseInt(10, desc="Number of cuts for each axis", usedefault=True)

    display_modes = traits.List(
        ['x', 'y', 'z'],
        usedefault=True,
        desc="Slicing axis to view",
        inner_traits=traits.Enum(values=['x', 'y', 'z']))


class _IAnatOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class IAnatRPT(reporting.ReportCapableInterface):

    input_spec = _IAnatInputSpecRPT
    output_spec = _IAnatOutputSpecRPT

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Instantiation of abstract method, does nothing

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime

    def _generate_report(self):

        data = nimg.load_img(self.inputs.nii)

        if len(data.shape) == 4:
            data = _make_3d_from_4d(data)

        bbox_nii = nimg.threshold_img(data, 1e-3)
        cuts = cuts_from_bbox(bbox_nii, cuts=self.inputs.n_cuts)
        robust_params = robust_set_limits(data.get_fdata().reshape(-1), {})

        svgs = []
        for d in self.inputs.display_modes:
            plot_params = {
                "display_mode": d,
                "cut_coords": cuts[d],
                **robust_params
            }
            display = nplot.plot_anat(data, **plot_params)
            svg = extract_svg(display)
            svg = svg.replace("figure_1", f"anatomical-{d}")
            svgs.append(fromstring(svg))
            display.close()

        compose_view(svgs, fg_svgs=None, out_file=self._out_report)


class _IFuncInputSpecRPT(nrc._SVGReportCapableInputSpec):
    nii = File(exists=True,
               usedefault=False,
               resolve=True,
               desc="Functional Image to view",
               mandatory=True)

    n_cuts = BaseInt(10, desc="Number of cuts for each axis", usedefault=True)

    display_modes = traits.List(
        ['x', 'y', 'z'],
        usedefault=True,
        desc="Slicing axis to view",
        inner_traits=traits.Enum(values=['x', 'y', 'z']))


class _IFuncOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class IFuncRPT(reporting.ReportCapableInterface):

    input_spec = _IFuncInputSpecRPT
    output_spec = _IFuncOutputSpecRPT

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Instantiation of abstract method, does nothing

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime

    def _generate_report(self):

        data = _make_3d_from_4d(nimg.load_img(self.inputs.nii))
        bbox_nii = nimg.threshold_img(data, 1e-3)
        cuts = cuts_from_bbox(bbox_nii, cuts=self.inputs.n_cuts)

        svgs = []
        for d in self.inputs.display_modes:
            plot_params = {"display_mode": d, "cut_coords": cuts[d]}
            display = nplot.plot_epi(data, **plot_params)
            svg = extract_svg(display)
            svg = svg.replace("figure_1", f"functional-{d}")
            svgs.append(fromstring(svg))
            display.close()

        compose_view(svgs, fg_svgs=None, out_file=self._out_report)


class _IRegInputSpecRPT(nrc._SVGReportCapableInputSpec):

    bg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Background NIFTI for SVG',
                  mandatory=True)

    fg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Foreground NIFTI for SVG',
                  mandatory=True)

    contours = File(exists=True,
                    usedefault=False,
                    resolve=True,
                    desc='Contours to include in image',
                    mandatory=False)


class _IRegOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class IRegRPT(nrc.RegistrationRC):
    """Implementation of Identity operation on RegistrationRC

    This class performs no operations and generates a report
    as a side-effect. It is primarily used to generate registration
    reports on already registered data.

    """

    input_spec = _IRegInputSpecRPT
    output_spec = _IRegOutputSpecRPT

    def _post_run_hook(self, runtime: Bunch) -> Bunch:
        """Side-effect function of IRegRPT.

        Generate transition report as a side-effect. No operations
        are performed on the data (identity)

        If a 4D image is passed in the first index will be pulled for viewing

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object propogated through ReportCapable
            interfaces

        """

        # Need to 3Dify 4D images and re-orient to RAS
        fi = _make_3d_from_4d(nimg.load_img(self.inputs.fg_nii))
        bi = _make_3d_from_4d(nimg.load_img(self.inputs.bg_nii))
        self._fixed_image = fi
        self._moving_image = bi

        return super(IRegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Main function of IRegRPT, does nothing.

        Implements identity operation. IRegRPT expects
        fully registered inputs, so no operations are performed.

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime


class _ISegInputSpecRPT(nrc._SVGReportCapableInputSpec):
    '''
    Input specification for ISegRPT, implements:

    anat_file: Input anatomical image
    seg_files: Input segmentation image(s) - can be a list or a single file
    mask_file: Input ROI mask

    Bases _SVGReportCapableInputSpec which implements:

    out_report: Filename trait
    compress_report: ["auto", true, false]

    '''
    anat_file = File(exists=True,
                     usedefault=False,
                     resolve=True,
                     desc='Anatomical image of SVG',
                     mandatory=True)

    seg_files = InputMultiPath(File(exists=True,
                                    usedefault=False,
                                    resolve=True),
                               desc='Segmentation image of SVG',
                               mandatory=True)

    mask_file = File(exists=True,
                     resolve=True,
                     desc='ROI Mask for mosaic',
                     mandatory=False)

    masked = traits.Bool(False,
                         usedefault=True,
                         desc='Flag to indicate whether'
                         ' image is already masked')


class _ISegOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class ISegRPT(nrc.SegmentationRC):
    '''
    Class to generate registration images from pre-existing
    NIFTI files.

    Effectively acts as an Identity node with report
    generation as a side-effect.
    '''

    # Use our declared IO specs
    input_spec = _ISegInputSpecRPT
    output_spec = _ISegOutputSpecRPT

    def _post_run_hook(self, runtime: Bunch) -> Bunch:
        """Side-effect function of ISegRPT.

        Generate transition report as a side-effect. No operations
        are performed on the data (identity)

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object propogated through ReportCapable
            interfaces

        """

        if not isinstance(self.inputs.seg_files, list):
            self.inputs.seg_files = [self.inputs.seg_files]

        # Set variables for `nrc.SegmentationRC`
        self._anat_file = self.inputs.anat_file
        self._seg_files = self.inputs.seg_files
        self._mask_file = self.inputs.mask_file or None
        self._masked = self.inputs.masked

        # Propogate to superclass
        return super(ISegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Main function of ISegRPT, does nothing.

        Implements identity operation. ISegRPT expects
        fully registered inputs, so no operations are performed.

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime


def _make_3d_from_4d(nii: Nifti1Image, ind: int = 0) -> Nifti1Image:
    '''
    Convert 4D Image into 3D one by pulling a single volume.
    Performs identity mapping if input image is 3D

    Args:
        nii: Input image
        ind: Index to pull from 4D image
    '''

    if len(nii.shape) < 4:
        return nii

    return nii.slicer[:, :, :, ind]


def _reorient_to_ras(img: Nifti1Image) -> Nifti1Image:
    '''
    Re-orient image to RAS

    Args:
        img: Image to re-orient to match ref image

    Returns:
        img re-oriented to RAS
    '''

    img = nimg.load_img(img)
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    img_ornt = nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(img.affine))
    img2ref = nib.orientations.ornt_transform(img_ornt, ras_ornt)
    return img.as_reoriented(img2ref)


# Register interfaces with adapter-factory
# TODO: Automate registration via using plugins pattern
def _run_imports() -> None:
    register_interface(IRegRPT, 'registration')
    register_interface(ISegRPT, 'segmentation')
    register_interface(IAnatRPT, 'anatomical')
    register_interface(IFuncRPT, 'functional')
