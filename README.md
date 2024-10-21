# Mars-Rover-Image-learning
Leverage NASA's website data scrapping for images by Rover on the surface of Mars. Finding areas for presence of water on the surface of the planet. 



# Mars Image Analysis for Potential Life Environments

![water on Mars](https://github.com/user-attachments/assets/02fc3e7c-d484-41b2-a291-63f1d0c1c85f)


## Project Overview
This project aims to analyze images of Mars to identify potential environments that could support life. Using various image processing techniques, we evaluated a large dataset of Mars images for features that might indicate habitable conditions or areas of scientific interest.

## Methodology
We developed a Python-based image analysis tool that processes Mars images and scores them based on four key features:
1. Dark Spot Ratio
2. Edge Ratio
3. Color Variation
4. Potential Water Features

Each image was assigned a composite score based on these features, allowing us to rank and identify the most promising images for further investigation.

## Key Components
- Image preprocessing and normalization
- Feature extraction (dark spots, edges, color variation, water-related features)
- Scoring algorithm
- Visualization of results

## Dataset
The dataset consists of various Mars images from different missions and instruments, including:
- Global views of Mars (e.g., PIA07443.jpg)
- Surface-level images (e.g., PIA08528.jpg)
- Spectral data visualizations (e.g., PIA05029.jpg)
- Elevation or composition maps (e.g., PIA04910.jpg)

![download (2)](https://github.com/user-attachments/assets/b5b76e38-4615-4f4a-a325-19450e7bf9f8)


## Results

### Top Scoring Images
1. PIA02818.jpg (Score: 0.3345)
2. PIA02820.jpg (Score: 0.3317)
3. PIA02819.jpg (Score: 0.3280)
4. PIA02816.jpg (Score: 0.3091)
5. PIA12861.jpg (Score: 0.3067)

### Key Findings
1. Diverse Mineral Deposits: The top-scoring images (PIA02818-PIA02816 series) showed moderate dark spot ratios, high color variation, and significant potential water features. These likely represent areas with diverse mineral compositions and possible water associations.

2. Potential Water/Ice Features: PIA12861.jpg stood out with very high potential water features (0.7413) in a bright, complex terrain, possibly indicating ice deposits or frost.

3. Unusual Terrains: Some high-scoring images (e.g., PIA18591.jpg) showed extreme characteristics, such as very high dark spot ratios combined with high edge ratios, representing unique Martian environments worthy of further study.

4. Correlation Patterns: The analysis revealed a general trend of decreasing edge ratio as dark spot ratio increases, with significant variations that could indicate interesting geological features.

## Visualization
A scatter plot was generated to visualize the relationship between dark spot ratio, edge ratio, color variation, and potential water features across all analyzed images. This plot highlighted clusters of images with similar characteristics and helped identify outliers of potential interest.

## Conclusions
1. The analysis successfully identified several areas of high interest for potential life-supporting environments on Mars.
2. Images with a balance of features (moderate dark spots, some edge complexity, high color variation, and potential water-related characteristics) scored highest in our analysis.
3. The diverse range of high-scoring images suggests multiple types of environments on Mars that could be of interest in the search for potential biosignatures or habitable conditions.

## Future Work
1. Detailed visual inspection of top-scoring images by experts.
2. Integration of additional data sources (e.g., spectral data, geological maps) to provide context for the image analysis results.
3. Refinement of the scoring algorithm based on expert feedback and additional scientific criteria.
4. Correlation of high-scoring areas with known regions of interest on Mars for validation and further investigation.

## Technologies Used
- Python
- NumPy
- PIL (Python Imaging Library)
- Matplotlib
- SciPy

## Acknowledgments
This project utilizes images from various Mars missions, including Mars Pathfinder, Mars Reconnaissance Orbiter, and others. All images are courtesy of NASA/JPL-Caltech.
https://pds-imaging.jpl.nasa.gov/search/?fq=TARGET_NAME%3Amars&fq=ATLAS_SPACECRAFT_NAME%3A%22carl%20sagan%20memorial%20station%22&fq=-ATLAS_THUMBNAIL_URL%3Abrwsnotavail.jpg&q=*%3A*&start=3120
