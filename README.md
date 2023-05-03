# Limitations of Physics-Informed deep learning in traffic State Estimation (PISE) 

### Corresponding publication 

  - A. J. Huang and S. Agarwal, ["On the Limitations of Physics-informed Deep Learning: Illustrations Using First Order Hyperbolic Conservation Law-based Traffic Flow Models"](https://ieeexplore.ieee.org/document/10105558) in *IEEE Open Journal of Intelligent Transportation Systems*, vol. 4, pp. 279-293, 2023, doi: 10.1109/OJITS.2023.3268026.
  
We wish to thank Dr. Rongye Shi at Columbia University for providing the ring road data for the first case study, Dr. Animesh Biswas at the University of Nebraska at Lincoln for the LWR reconstruction of the data, and Dr. Pushkin Kachroo at the University of Nevada Las Vegas for the suggestions and discussions on this topic.

### Citation

    @article{huang2023limitations,
      title={On the Limitations of Physics-informed Deep Learning: Illustrations   Using First Order Hyperbolic Conservation Law-based Traffic Flow Models},
      author={Huang, Archie J and Agarwal, Shaurya},
      journal={IEEE Open Journal of Intelligent Transportation Systems},
      year={2023},
      volume={4},
      number={},
      pages={279-293},
      doi={10.1109/OJITS.2023.3268026}
      publisher={IEEE}
    }

### Reference

Code is built upon Dr. Maziar Raissi's PINNs - https://github.com/maziarraissi/PINNs 

- Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." Journal of Computational Physics 378 (2019): 686-707.

Field Data in this repo -       

NGSIM_US101_Density_Data.txt  - Vehicle Density on US-101 Highway Segment, between 7:50 am and 8:35 am, NGSIM. (Source: Dr. Allan Avila - https://github.com/Allan-Avila/Highway-Traffic-Dynamics-KMD-Code)
