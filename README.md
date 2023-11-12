# DRIVER

This is the code for paper *Live Streaming recommendations based on dynamic representation learning* on *Decision Support Systems*.

Due to a non-disclosure agreement and potential private issues, we cannot disclose the original dataset used in the paper.
We randomly select 10% data samples to show the workflow of DRIVER. It is stressed that the codes just show how DRIVER works
but not the performance.

Please add this citation if you use our code or method for academic purpose:

@article{gao2023live,
title = {Live streaming recommendations based on dynamic representation learning},
journal = {Decision Support Systems},
volume = {169},
pages = {113957},
year = {2023},
issn = {0167-9236},
doi = {https://doi.org/10.1016/j.dss.2023.113957},
url = {https://www.sciencedirect.com/science/article/pii/S0167923623000325},
author = {Ge Gao and Hongyan Liu and Kang Zhao},
keywords = {Machine learning, Design science, Recommender systems, Consumer path},
abstract = {As an emerging form of social media, live streaming services (e.g., Twitch and Clubhouse) allow users to interact with hosts and peers in real time while enjoying shows or participating in discussions. These platforms are also dynamic, with shows or discussions changing quickly inside a room and users frequently switching between rooms. To improve user engagement and experience on such platforms, we design a new recommendation model named Dynamic Representations for Live Streaming Rooms (DRIVER) to provide room recommendations. Guided by the Integrated Framework for Consumer Path Modeling and the social affordance theory, DRIVER infers dynamic representations of live streaming rooms by leveraging users’ behavior paths in entering, staying in, and leaving rooms. One contribution of our model is a new and efficient dynamic learning framework to model instantaneous and ever-changing inter-room relationships by considering individual users’ behavior paths after leaving a room. Also supported by social affordance theory, another methodological novelty of our model is to capture dynamic characteristics of a room by incorporating features of the current audience inside the room. Experiments on real-world datasets from two different types of live streaming platforms demonstrate that DRIVER outperforms state-of-the-art representation learning methods and sequential recommender systems. The proposed method also has implications for recommender system design in other contexts, in which items are characterized by users’ dynamic behavior paths and ongoing social interactions.}
}

