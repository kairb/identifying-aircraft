# Identifying Aircraft from Above
We all know what an aircraft looks like, but does a computer? A seemingly simple task that can be carried out by individuals at age two, poses a complex problem to modern technology. Machine learning is a relatively new field with little research but already boasts claim to many applications such as driverless cars and face recognition systems. The development of object recognition is the center of many companies’ business models and objectives making aircraft identification such an interesting topic to research.

Existing images of ground and aircraft are pre-processed using Histogram of Gradients to create feature descriptors. Feature descriptors describe the orientation of a gradient within an image subsection. Support vector machines are passed feature descriptors with labels for training. Once training is completed, the support vector machine accepts a test set and returns predictions. Large image search takes a large image and looks within a smaller area for aircraft. Search area parameters are provided by the user.

The results obtained from cross validation show an accuracy of 100% when identifying standalone aircraft. However, when searching for aircraft in larger images, accuracy drastically decreases to around 55% some aircraft are overlooked. After optimization, the system used to identify aircraft can be applied to other identification problems with possible military and commercial uses. The software can be given any set data meaning there’s no theoretical limit to what it can and can’t recognize.

## Getting Started (Software Based)
To run my program, pull the entire project from gitlab and follow the install instructions. There is no built version of the application yet as the software is still in design stage

### Prerequisites
To enable this project to run correctly, you must have the following:
* Pyhton 3.5 (or later)
* TensorFlow
* Keras
* Scikit-learn
* Scikit-Image
* opencv-pyhton
* fpdf


#### Install libraries and run program
```
run requirements.bat to install required dependencies

then
main.py to start GUI

```

### Running Tests
Cross validation test 10 fold validation of training data

### Versioning Statergy
This proect makes use of semantic versioning (https://semver.org/).


## Authors
* Kai Roper-Blackman

## References
* [Gitlab Markdown Guide](https://docs.gitlab.com/ee/user/markdown.html)
