# Datasets

## Scope

The study evaluates SwinEADFormer on three publicly available remote sensing
change detection datasets: LEVIR-CD, WHU-CD, and SYSU-CD. No new remote sensing
image dataset was generated.

This repository does not contain dataset files, download scripts, preprocessing
code, split files, or derived image patches. Dataset access remains under the
control of the original providers.

## Dataset overview

| Dataset | Main task scope | Public data form | Spatial resolution | Location and acquisition period | Official source |
|---|---|---:|---:|---|---|
| LEVIR-CD | Building change detection | 637 bitemporal image pairs of 1024 x 1024 pixels | 0.5 m/pixel | Multiple regions in Texas, USA; image pairs span 5-14 years | [LEVIR-CD project page](https://justchenhao.github.io/LEVIR/) |
| WHU-CD | Building change detection | Two large aerial images with building and change labels | 0.2 m/pixel in the processed form used in the study; original aerial imagery is 0.075 m/pixel | Christchurch, New Zealand; April 2012 and 2016 | [WHU building dataset page](https://gpcv.whu.edu.cn/data/building_dataset.html) |
| SYSU-CD | General binary land-cover change detection | 20,000 bitemporal image pairs of 256 x 256 pixels | 0.5 m/pixel | Hong Kong, China; 2007-2014 | [SYSU-CD repository](https://github.com/liumency/SYSU-CD) |

The descriptions above summarize the information provided by the original
dataset maintainers. Users should consult the official sources for the current
download locations, file organization, labels, and provider terms.

## LEVIR-CD

LEVIR-CD is a large-scale building change detection dataset constructed from
Google Earth imagery. It contains 637 pairs of high-resolution images, each
with a size of 1024 x 1024 pixels and a spatial resolution of 0.5 m/pixel. The
image pairs cover multiple regions in Texas and have acquisition intervals of
5 to 14 years. The provider reports 31,333 annotated building-change
instances.

The dataset is building-focused. Its binary labels represent changed and
unchanged pixels. The original full-resolution images are divided into smaller
patches for model training and evaluation in this study.

Original publication:

> H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New
> Dataset for Remote Sensing Image Change Detection," *Remote Sensing*, vol.
> 12, no. 10, article 1662, 2020.
> [https://doi.org/10.3390/rs12101662](https://doi.org/10.3390/rs12101662)

## WHU-CD

WHU-CD is the building change detection subset of the WHU building dataset. It
covers approximately 20.5 square kilometres in Christchurch, New Zealand. The
two aerial images were acquired in April 2012 and 2016. The provider reports
12,796 buildings in the 2012 image and 16,077 buildings in the 2016 image. The
images were geo-rectified by the provider, with a reported registration
accuracy of 1.6 pixels.

The provider reports an original aerial-image ground resolution of 0.075 m and
also distributes downsampled raster products. The processed form used in this
study has a spatial resolution of 0.2 m/pixel.

The source images and labels are divided into non-overlapping 256 x 256 patches
for the protocol used in this study. Small residual registration errors may
still affect pixels near object boundaries and should be considered when
interpreting boundary-level errors.

Original publication:

> S. Ji, S. Wei, and M. Lu, "Fully Convolutional Networks for Multi-Source
> Building Extraction from an Open Aerial and Satellite Imagery Dataset,"
> *IEEE Transactions on Geoscience and Remote Sensing*, vol. 57, no. 1,
> pp. 574-586, 2019.
> [https://doi.org/10.1109/TGRS.2018.2858817](https://doi.org/10.1109/TGRS.2018.2858817)

## SYSU-CD

SYSU-CD contains 20,000 pairs of 256 x 256 aerial images at a spatial
resolution of 0.5 m/pixel. The images cover Hong Kong and were acquired between
2007 and 2014. Its annotations include several forms of change, such as newly
constructed buildings, suburban expansion, groundwork before construction,
vegetation change, road expansion, and coastal construction.

SYSU-CD is broader than a building-only benchmark. Results on this dataset
therefore reflect general binary change detection performance across multiple
change types, not only completed building changes.

Original publication:

> Q. Shi, M. Liu, S. Li, X. Liu, F. Wang, and L. Zhang, "A Deeply Supervised
> Attention Metric-Based Network and an Open Aerial Image Dataset for Remote
> Sensing Change Detection," *IEEE Transactions on Geoscience and Remote
> Sensing*, vol. 60, article 5604816, 2022.
> [https://doi.org/10.1109/TGRS.2021.3085870](https://doi.org/10.1109/TGRS.2021.3085870)

## Protocol used in this study

All three datasets are evaluated using RGB image pairs and binary change
labels. The model input size is 256 x 256 pixels. The patch-level split and
changed-pixel statistics used in the manuscript are:

| Dataset | Training patches | Validation patches | Test patches | Total patches | Changed-pixel ratio |
|---|---:|---:|---:|---:|---:|
| LEVIR-CD | 7,120 | 1,024 | 2,048 | 10,192 | 4.65% |
| WHU-CD | 1,134 | 126 | 690 | 1,950 | 5.96% |
| SYSU-CD | 12,000 | 4,000 | 4,000 | 20,000 | 21.83% |

These are the patch-level statistics of the protocol used in this study. They
should not be interpreted as new dataset releases or as replacements for the
official dataset descriptions. This repository provides the model, dataset
loader, training and evaluation entry points, loss functions, and metric
implementations. It does not redistribute the original datasets, derived image
patches, or dataset split manifests.

The changed-pixel ratios show that the three benchmarks have different levels
of class imbalance. LEVIR-CD and WHU-CD are strongly dominated by unchanged
pixels. SYSU-CD contains a higher proportion of changed pixels and a broader
set of semantic change categories. Cross-dataset metric comparisons should
therefore be interpreted with care.

## Access, terms, and redistribution

- Obtain each dataset from its original provider through the official links
  above.
- The LEVIR-CD provider limits the images and annotations to academic,
  non-commercial use and requires compliance with the Google Earth terms of
  use.
- The cited WHU-CD page and SYSU-CD repository are the authoritative sources
  for their current access and use conditions. No separate dataset license is
  asserted by this repository.
- Review and comply with the terms, citation requirements, and permitted uses
  specified by each provider.
- Do not assume that the license of this source-code repository applies to any
  dataset.
- This repository does not grant rights to redistribute the original images,
  labels, or derived patches.
- Dataset availability, download URLs, and provider terms may change. The
  original provider is the authoritative source.

## Manuscript Data Availability statement

The following text can be used in the manuscript:

> The publicly available LEVIR-CD, WHU-CD, and SYSU-CD datasets analyzed in
> this study can be obtained from their original providers at
> https://justchenhao.github.io/LEVIR/,
> https://gpcv.whu.edu.cn/data/building_dataset.html, and
> https://github.com/liumency/SYSU-CD, respectively. No new remote sensing
> image dataset was generated. The public implementation of
> SwinEADFormer is available at https://github.com/cujoh777/EAD-Former. The
> repository does not redistribute the original datasets, derived image
> patches, or dataset split manifests.
