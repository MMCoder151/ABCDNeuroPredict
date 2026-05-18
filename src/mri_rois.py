
def mri_rois():
    # Define tabular MRI data files to import
    mri_files = ["mr_y_smri__vol__dst.tsv", # Structural MRI - Volumes (Destrieux) [Youth]
                "mr_y_smri__thk__dst.tsv", # Structural MRI - Cortical Thickness (Destrieux) [Youth]
                "mr_y_dti__fs__fa__at.tsv", # DTI (Full shell) - Fractional Anisotropy (AtlasTrack) [Youth]
                "mr_y_rsfmri__corr__gpnet.tsv", # Resting State fMRI - Correlations (Gordon network) [Youth] 
                "mr_y_rsfmri__corr__gpnet__aseg.tsv" # Resting State fMRI - Correlations (Gordon network to Subcortical) [Youth]
                ] # Add more files as needed

    # Define MRI ROIs as dictionary with variable names and descriptions
    mri_rois = {
        # Structural MRI - Volumes (Destrieux) [Youth]
        "mr_y_smri__vol__dst__gsca__lh_sum":"Total volume of Destrieux ROI: Anterior part of the cingulate gyrus and sulcus (Left hemisphere)",
        "mr_y_smri__vol__dst__gilsci__rh_sum":"Total volume of Destrieux ROI: Long insular gyrus and central sulcus of the insula (Right hemisphere)",
        "mr_y_smri__vol__dst__gis__rh_sum":"Total volume of Destrieux ROI: Short insular gyri (Right hemisphere)",
        "mr_y_smri__vol__dst__gilsci__lh_sum":"Total volume of Destrieux ROI: Long insular gyrus and central sulcus of the insula (Left hemisphere)",
        "mr_y_smri__vol__dst__gis__lh_sum":"Total volume of Destrieux ROI: Short insular gyri (Left hemisphere)",
        "mr_y_smri__vol__dst__gfs__rh_sum":"Total volume of Destrieux ROI: Superior frontal gyrus (Right hemisphere)",
        "mr_y_smri__vol__dst__gfs__lh_sum":"Total volume of Destrieux ROI: Superior frontal gyrus (Left hemisphere)",
        # Structural MRI - Cortical Thickness (Destrieux) [Youth]
        "mr_y_smri__thk__dst__gsca__rh_mean":"Average cortical thickness of Destrieux ROI: Anterior part of the cingulate gyrus and sulcus (Right hemisphere)",
        "mr_y_smri__thk__dst__gsca__lh_mean":"Average cortical thickness of Destrieux ROI: Anterior part of the cingulate gyrus and sulcus (Left hemisphere)",
        "mr_y_smri__thk__dst__gilsci__rh_mean":"Average cortical thickness of Destrieux ROI: Long insular gyrus and central sulcus of the insula (Right hemisphere)",
        "mr_y_smri__thk__dst__gis__rh_mean":"Average cortical thickness of Destrieux ROI: Short insular gyri (Right hemisphere)",
        "mr_y_smri__thk__dst__gilsci__lh_mean":"Average cortical thickness of Destrieux ROI: Long insular gyrus and central sulcus of the insula (Left hemisphere)",
        "mr_y_smri__thk__dst__gis__lh_mean":"Average cortical thickness of Destrieux ROI: Short insular gyri (Left hemisphere)",
        "mr_y_smri__thk__dst__go__rh_mean":"Average cortical thickness in Destrieux ROI: Orbital gyri (Right hemisphere)",
        "mr_y_smri__thk__dst__gr__rh_mean":"Average cortical thickness in Destrieux ROI: Rectus gyrus (Right hemisphere)",
        "mr_y_smri__thk__dst__go__lh_mean":"Average cortical thickness in Destrieux ROI: Orbital gyri (Left hemisphere)",
        "mr_y_smri__thk__dst__gr__lh_mean":"Average cortical thickness in Destrieux ROI: Rectus gyrus (Left hemisphere)",
        # DTI (Full shell) - Fractional Anisotropy (AtlasTrack) [Youth]
        "mr_y_dti__fs__fa__atmr_y_dti__fs__fa__at__cc_wmean":"Weighted average fractional anisotropy (Full shell DTI) in AtlasTrack fiber tract: Corpus callosum",
        "mr_y_dti__fs__fa__at__atr__lh_wmean":"Weighted average fractional anisotropy (Full shell DTI) in AtlasTrack fiber tract: Anterior thalamic radiation (left hemisphere)",
        "mr_y_dti__fs__fa__at__cst__lh_wmean":"Weighted average fractional anisotropy (Full shell DTI) in AtlasTrack fiber tract: Corticospinal tract (left hemisphere)",
        "mr_y_dti__fs__fa__at__ifo__rh_wmean":"Weighted average fractional anisotropy (Full shell DTI) in AtlasTrack fiber tract: Inferior fronto-occipital fasciculus (right hemisphere)",
        "mr_y_dti__fs__fa__at__cgc__rh_wmean":"Weighted average fractional anisotropy (Full shell DTI) in AtlasTrack fiber tract: Cingulate cingulum (Right hemisphere)",
        "mr_y_dti__fs__fa__at__cgh__rh_wmean":"Weighted average fractional anisotropy (Full shell DTI) in AtlasTrack fiber tract: Parahpcm cingulum (Right hemisphere)",
        # Resting State fMRI - Correlations (Gordon network) [Youth]
        "mr_y_rsfmri__corr__gpnet__def__def_mean":"Average correlation between Gordon networks: Default & default",
        "mr_y_rsfmri__corr__gpnet__def__doa_mean":"Average correlation between Gordon networks: Default & dorsal attention",
        "mr_y_rsfmri__corr__gpnet__def__vea_mean":"Average correlation between Gordon networks: Default & ventral attention",
        "mr_y_rsfmri__corr__gpnet__def__sal_mean":"Average correlation between Gordon networks: Default & salience",
        "mr_y_rsfmri__corr__gpnet__doa__doa_mean":"Average correlation between Gordon networks: Dorsal attention & dorsal attention",
        "mr_y_rsfmri__corr__gpnet__doa__sal_mean":"Average correlation between Gordon networks: Dorsal attention & salience",
        "mr_y_rsfmri__corr__gpnet__doa__vea_mean":"Average correlation between Gordon networks: Dorsal attention & ventral attention",
        "mr_y_rsfmri__corr__gpnet__doa__frp_mean":"Average correlation between Gordon networks: Dorsal attention & frontoparietal",
        "mr_y_rsfmri__corr__gpnet__doa__smh_mean":"Average correlation between Gordon networks: Dorsal attention & somatomotor hand",
        "mr_y_rsfmri__corr__gpnet__doa__smm_mean":"Average correlation between Gordon networks: Dorsal attention & somatomotor mouth",
        "mr_y_rsfmri__corr__gpnet__vea__vea_mean":"Average correlation between Gordon networks: Ventral attention & ventral attention",
        # Resting State fMRI - Correlations (Gordon network to Subcortical) [Youth]
        "mr_y_rsfmri__corr__gpnet__aseg__doa__hc__lh_mean":"Average correlation between Gordon network: Dorsal attention & Subcortical ROI: Hippocampus (Left hemisphere)",
        "mr_y_rsfmri__corr__gpnet__aseg__smh__ag__lh_mean":"Average correlation between Gordon network: Sensorimotor hand & Subcortical ROI: Amygdala (Left hemisphere)",
        "mr_y_rsfmri__corr__gpnet__aseg__smm__ag__lh_mean":"Average correlation between Gordon network: Sensorimotor mouth & Subcortical ROI: Amygdala (Left hemisphere)",
        "mr_y_rsfmri__corr__gpnet__aseg__sal__ag__lh_mean":"Average correlation between Gordon network: Salience & Subcortical ROI: Amygdala (Left hemisphere)",
        "mr_y_rsfmri__corr__gpnet__aseg__doa__hc__rh_mean":"Average correlation between Gordon network: Dorsal attention & Subcortical ROI: Hippocampus (Right hemisphere)",
        "mr_y_rsfmri__corr__gpnet__aseg__smh__ag__rh_mean":"Average correlation between Gordon network: Sensorimotor hand & Subcortical ROI: Amygdala (Right hemisphere)",
        "mr_y_rsfmri__corr__gpnet__aseg__smm__ag__rh_mean":"Average correlation between Gordon network: Sensorimotor mouth & Subcortical ROI: Amygdala (Right hemisphere)",
        "mr_y_rsfmri__corr__gpnet__aseg__sal__ag__rh_mean":"Average correlation between Gordon network: Salience & Subcortical ROI: Amygdala (Right hemisphere)"
    }
    return mri_files, mri_rois