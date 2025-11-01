from app import db
from app.models import User, Subject, Quiz, QuizQuestion, QuizScore
from datetime import datetime

USERS_DATA = [
    {
        "id": 1,
        "username": "brian@gmail.com",
        "password": "password",
        "fullname": "Brian Rainer",
        "nickname": "Brian",
        "is_admin": True
    },
    {
        "id": 2,
        "username": "john@quizland.com",
        "password": "superman123",
        "fullname": "John Smith",
        "nickname": "John",
        "is_admin": False
    },
    {
        "id": 3,
        "username": "jane@quizland.com",
        "password": "ilovebanana",
        "fullname": "Jane Doe",
        "nickname": "Jane",
        "is_admin": False
    }
]

SUBJECT_DATA = [
    {
        "id": 1,
        "name": "Computer Vision",
        "description": "Learn all about how computer parse graphical information"
    },
    {
        "id": 2,
        "name": "Artificial Intelligence with Python",
        "description": "Learn how to build AI using python"
    },
    {
        "id": 3,
        "name": "Neuro-Linguistic Programming",
        "description": "Learn about NLP and its application"
    }
]

QUIZ_DATA = [
    {
        "id": 1,
        "name": "Computer Vision Quiz",
        "description": "20 Questions to test your knowledge about computer vision",
        "date_of_quiz": datetime(2025,11,1),
        "duration": 7200,
        "subject_id": 1
    },
    {
        "id": 2,
        "name": "AI with Python Quiz",
        "description": "20 Questions to test your knowledge about artificial intelligence",
        "date_of_quiz": datetime(2025,11,1),
        "duration": 7200,
        "subject_id": 2
    },
    {
        "id": 3,
        "name": "NLP Quiz",
        "description": "20 Questions to test your knowledge about neuro-linguistic programming",
        "date_of_quiz": datetime(2025,11,1),
        "duration": 7200,
        "subject_id": 3
    }
]

QUIZ_COMPUTER_VISION_QUESTIONS = [
    {
      "id": 1,
      "question": "What is the fundamental unit of a digital image, representing a single point in the image with its corresponding color and intensity value?",
      "options": ["Raster", "Voxel", "Pixel", "Kernel"],
      "correct_answer": "Pixel"
    },
    {
      "id": 2,
      "question": "In image processing, what mathematical operation is performed when a small matrix (a kernel or filter) is swept across an entire image to produce a new modified image?",
      "options": ["Fourier Transform", "Hough Transform", "Normalization", "Convolution"],
      "correct_answer": "Convolution"
    },
    {
      "id": 3,
      "question": "Which classic algorithm for edge detection is known for using two different threshold values to link weak and strong edges, a process called hysteresis?",
      "options": ["Sobel Operator", "Canny Edge Detector", "Laplacian of Gaussian (LoG)", "Harris Corner Detector"],
      "correct_answer": "Canny Edge Detector"
    },
    {
      "id": 4,
      "question": "Which local feature descriptor is celebrated for being invariant to changes in scale, rotation, and illumination, making it robust for matching objects across different viewpoints?",
      "options": ["LBP (Local Binary Patterns)", "HOG (Histogram of Oriented Gradients)", "SIFT (Scale-Invariant Feature Transform)", "Haar-like Features"],
      "correct_answer": "SIFT (Scale-Invariant Feature Transform)"
    },
    {
      "id": 5,
      "question": "What is the primary role of a **Pooling Layer** in a Convolutional Neural Network (CNN)?",
      "options": ["To introduce non-linearity into the model's output.", "To extract high-level feature maps using learnable filters.", "To reduce the spatial size of the representation, reducing computation and making features more robust.", "To classify the image based on the extracted features."],
      "correct_answer": "To reduce the spatial size of the representation, reducing computation and making features more robust."
    },
    {
      "id": 6,
      "question": "Which computer vision task involves classifying every single pixel in an image into a specific category (e.g., car, road, sky), creating a dense, per-pixel mask?",
      "options": ["Object Detection", "Image Classification", "Instance Segmentation", "Semantic Segmentation"],
      "correct_answer": "Semantic Segmentation"
    },
    {
      "id": 7,
      "question": "What is the name of the technique used to adjust the intensity values of pixels in an image so that the image's overall contrast is enhanced, by distributing the most frequent intensity values?",
      "options": ["Homomorphic Filtering", "Gamma Correction", "Gaussian Blurring", "Histogram Equalization"],
      "correct_answer": "Histogram Equalization"
    },
    {
      "id": 8,
      "question": "Which activation function is most widely used in the hidden layers of modern deep Convolutional Neural Networks (CNNs) due to its computational efficiency and its role in mitigating the vanishing gradient problem?",
      "options": ["Sigmoid", "Softmax", "Hyperbolic Tangent (Tanh)", "ReLU (Rectified Linear Unit)"],
      "correct_answer": "ReLU (Rectified Linear Unit)"
    },
    {
      "id": 9,
      "question": "In the context of 3D computer vision, what technique involves using two cameras that are horizontally offset from each other to calculate the depth of points in a scene by measuring the displacement between corresponding points?",
      "options": ["Structure from Motion (SfM)", "Photometric Stereo", "Monocular Depth Estimation", "Stereo Vision"],
      "correct_answer": "Stereo Vision"
    },
    {
      "id": 10,
      "question": "Which family of object detection algorithms is known for being a 'single-shot' detector that treats object detection as a simple regression problem, directly predicting bounding boxes and class probabilities from full images in a single evaluation?",
      "options": ["R-CNN", "SSD (Single Shot MultiBox Detector)", "Faster R-CNN", "YOLO (You Only Look Once)"],
      "correct_answer": "YOLO (You Only Look Once)"
    },
    {
      "id": 11,
      "question": "In a typical CNN architecture, the **Fully Connected Layer** is usually found at the end of the network. What is its main function?",
      "options": ["To reduce the feature map's resolution by downsampling.", "To apply learnable filters to local regions of the input volume.", "To use the high-level features learned by previous layers for the final classification decision.", "To introduce non-linearity into the feature space."],
      "correct_answer": "To use the high-level features learned by previous layers for the final classification decision."
    },
    {
      "id": 12,
      "question": "Which geometric image transformation involves a shear mapping, such that it preserves straight lines and parallelism, but does not necessarily preserve distances, angles, or ratios of lengths?",
      "options": ["Euclidean Transformation", "Perspective Transformation", "Similarity Transformation", "Affine Transformation"],
      "correct_answer": "Affine Transformation"
    },
    {
      "id": 13,
      "question": "What is the computer vision technique used to estimate the apparent motion of brightness patterns in an image sequence (i.e., a video), often used for motion tracking or video stabilization?",
      "options": ["Background Subtraction", "Structure from Motion", "Optical Flow", "Non-Maximum Suppression"],
      "correct_answer": "Optical Flow"
    },
    {
      "id": 14,
      "question": "Which type of noise is characterized by random, isolated bright and dark pixels in an image, often caused by sudden, sharp disturbances?",
      "options": ["Gaussian Noise", "Quantization Noise", "Periodic Noise", "Salt-and-Pepper Noise"],
      "correct_answer": "Salt-and-Pepper Noise"
    },
    {
      "id": 15,
      "question": "Which non-linear filter is generally considered most effective for reducing 'salt-and-pepper' noise while preserving the sharpness of edges in the image?",
      "options": ["Mean (Average) Filter", "Gaussian Filter", "Laplacian Filter", "Median Filter"],
      "correct_answer": "Median Filter"
    },
    {
      "id": 16,
      "question": "In the evolution of Region-based CNNs (R-CNN, Fast R-CNN, Faster R-CNN), what was the key innovation of Fast R-CNN that significantly sped up training and testing compared to the original R-CNN?",
      "options": ["The introduction of the Region Proposal Network (RPN).", "The use of a single-shot regression model to eliminate region proposals.", "Passing the full image through the CNN once and using a 'Region of Interest' pooling layer.", "Using an all-new architecture without convolutional layers."],
      "correct_answer": "Passing the full image through the CNN once and using a 'Region of Interest' pooling layer."
    },
    {
      "id": 17,
      "question": "Which color space is often used in computer vision for skin detection or video processing because it effectively separates the image's luminance (brightness) from its chrominance (color information)?",
      "options": ["RGB (Red, Green, Blue)", "CMYK (Cyan, Magenta, Yellow, Key/Black)", "HSV (Hue, Saturation, Value)", "XYZ (CIE 1931 Color Space)"],
      "correct_answer": "HSV (Hue, Saturation, Value)"
    },
    {
      "id": 18,
      "question": "The **Hough Transform** is a technique frequently used in computer vision. What is the primary geometric element it is most commonly used to detect in an image?",
      "options": ["Blobs (Regions of interest with uniform color)", "Corners (Intersections of two or more edges)", "Curved Surfaces (3D structure)", "Lines and Circles"],
      "correct_answer": "Lines and Circles"
    },
    {
      "id": 19,
      "question": "In the training of a neural network for image classification, what is the primary role of the **Backpropagation** algorithm?",
      "options": ["To perform a forward pass and calculate the final prediction of the network.", "To calculate the gradient of the loss function with respect to every weight in the network.", "To apply the calculated gradients to update the model's weights.", "To reduce the computational complexity by eliminating redundant neurons."],
      "correct_answer": "To calculate the gradient of the loss function with respect to every weight in the network."
    },
    {
      "id": 20,
      "question": "A researcher is training a vision model using a large dataset of images where each image is paired with a corresponding label (e.g., 'cat', 'dog', 'car'). This approach falls under which machine learning paradigm?",
      "options": ["Reinforcement Learning", "Unsupervised Learning", "Supervised Learning", "Self-Supervised Learning"],
      "correct_answer": "Supervised Learning"
    }
]

QUIZ_DATA_AI_PYTHON_QUESTIONS = [
    {
      "id": 21,
      "question": "Which Python library is fundamental for high-performance numerical operations, especially vector and matrix manipulation, crucial for efficient AI computation?",
      "options": ["Pandas", "Matplotlib", "NumPy", "SciPy"],
      "correct_answer": "NumPy"
    },
    {
      "id": 22,
      "question": "In the Pandas library, what is the most common function used to handle missing values by filling them with a specified value (like the mean or median)?",
      "options": ["df.drop()", "df.replace()", "df.fillna()", "df.impute()"],
      "correct_answer": "df.fillna()"
    },
    {
      "id": 23,
      "question": "Which specific function within the scikit-learn module is used to split a dataset into training and testing subsets?",
      "options": ["train_test_split", "data_splitter", "model_selector", "cross_validate"],
      "correct_answer": "train_test_split"
    },
    {
      "id": 24,
      "question": "When defining an Artificial Neural Network in Keras, which layer type is used to connect every input node to every output node, often placed near the output layer?",
      "options": ["Convolutional", "Pooling", "Dropout", "Dense"],
      "correct_answer": "Dense"
    },
    {
      "id": 25,
      "question": "What is the high-level API for building and training deep learning models, often integrated into TensorFlow, known for its user-friendliness and rapid prototyping?",
      "options": ["PyTorch Lightning", "Scikit-learn", "Keras", "Theano"],
      "correct_answer": "Keras"
    },
    {
      "id": 26,
      "question": "In deep learning optimization using Python, which parameter controls the size of the step taken in the direction of the negative gradient, influencing how quickly or slowly the model learns?",
      "options": ["Batch Size", "Epoch", "Momentum", "Learning Rate"],
      "correct_answer": "Learning Rate"
    },
    {
      "id": 27,
      "question": "What is the primary purpose of the **validation set** during the neural network training loop?",
      "options": ["To tune the final model hyperparameters before deployment.", "To train the model's weights and biases.", "To provide an unbiased evaluation of the final model.", "To monitor model performance and prevent overfitting during training."],
      "correct_answer": "To monitor model performance and prevent overfitting during training."
    },
    {
      "id": 28,
      "question": "Which unsupervised learning algorithm, implemented in `sklearn.cluster`, requires the user to pre-define the number of groups (k)?",
      "options": ["DBSCAN", "Isolation Forest", "K-Means", "Principal Component Analysis"],
      "correct_answer": "K-Means"
    },
    {
      "id": 29,
      "question": "When using Pandas for feature engineering, which method is typically used to convert categorical string data (like 'Red', 'Blue') into numerical columns (e.g., 0s and 1s)?",
      "options": ["df.map()", "df.apply()", "pd.get_dummies()", "df.astype(int)"],
      "correct_answer": "pd.get_dummies()"
    },
    {
      "id": 30,
      "question": "Which Python library is commonly used for creating static, interactive, and animated visualizations, essential for exploring and communicating AI data insights?",
      "options": ["Seaborn", "Plotly", "Matplotlib", "Bokeh"],
      "correct_answer": "Matplotlib"
    },
    {
      "id": 31,
      "question": "In scikit-learn's linear models (like Lasso or Ridge), what is the main goal of adding a regularization term to the loss function?",
      "options": ["To speed up the training process.", "To increase the model's complexity.", "To prevent overfitting by penalizing large coefficients.", "To handle missing data points."],
      "correct_answer": "To prevent overfitting by penalizing large coefficients."
    },
    {
      "id": 32,
      "question": "Which Python library is most focused on natural language processing (NLP), providing tools for tokenization, stemming, and classification?",
      "options": ["Pandas", "Scikit-learn", "NLTK (Natural Language Toolkit)", "OpenCV"],
      "correct_answer": "NLTK (Natural Language Toolkit)"
    },
    {
      "id": 33,
      "question": "Which type of neural network is primarily used in Python for tasks involving sequential data, such as time series prediction or natural language translation?",
      "options": ["CNN (Convolutional Neural Network)", "Autoencoder", "Fully Connected Network", "RNN (Recurrent Neural Network)"],
      "correct_answer": "RNN (Recurrent Neural Network)"
    },
    {
      "id": 34,
      "question": "What is a **Tensor** in the context of TensorFlow or PyTorch?",
      "options": ["A simple Python list used for storing data.", "A variable containing the learning rate.", "A specific type of activation function.", "The fundamental data structure (multi-dimensional array) used for all operations."],
      "correct_answer": "The fundamental data structure (multi-dimensional array) used for all operations."
    },
    {
      "id": 35,
      "question": "Which Python module is most commonly used to save and load trained machine learning models (e.g., scikit-learn or Keras models) to disk for later use?",
      "options": ["csv", "json", "pickle / joblib", "io"],
      "correct_answer": "pickle / joblib"
    },
    {
      "id": 36,
      "question": "What is the purpose of **k-fold cross-validation** in scikit-learn?",
      "options": ["To train the model once on the entire dataset.", "To select the best features for the model.", "To obtain a more robust estimate of the model's performance by training and testing on different subsets.", "To scale the features to a common range."],
      "correct_answer": "To obtain a more robust estimate of the model's performance by training and testing on different subsets."
    },
    {
      "id": 37,
      "question": "Which metric measures the proportion of *positive* predictions that were *actually correct*, answering the question: 'Of all the times we said yes, how many times were we right?'",
      "options": ["Recall (Sensitivity)", "F1-Score", "Specificity", "Precision"],
      "correct_answer": "Precision"
    },
    {
      "id": 38,
      "question": "Which scikit-learn technique is used to reduce the number of features in a dataset while retaining most of the variance by projecting the data onto a lower-dimensional subspace?",
      "options": ["Linear Regression", "K-Means", "PCA (Principal Component Analysis)", "Gradient Boosting"],
      "correct_answer": "PCA (Principal Component Analysis)"
    },
    {
      "id": 39,
      "question": "In the context of supervised learning with Python, what does the term **'epoch'** refer to in model training?",
      "options": ["The size of the batch of data processed at once.", "A single forward and backward pass of the entire training dataset.", "The total number of parameters in the network.", "The process of monitoring the validation loss."],
      "correct_answer": "A single forward and backward pass of the entire training dataset."
    },
    {
      "id": 40,
      "question": "Which Python library is the standard open-source tool for real-time computer vision tasks like image manipulation, video analysis, and object tracking, often imported as `cv2`?",
      "options": ["Scikit-image", "PIL (Pillow)", "OpenCV (cv2)", "Matplotlib"],
      "correct_answer": "OpenCV (cv2)"
    }
]

QUIZ_DATA_NLP = [
    {
      "id": 41,
      "question": "In NLP, what is the term for the process of breaking a text stream into smaller meaningful units, such as words, punctuation, or subwords?",
      "options": ["Lemmatization", "Stemming", "Tokenization", "Parsing"],
      "correct_answer": "Tokenization"
    },
    {
      "id": 42,
      "question": "What common text pre-processing step involves reducing an inflected word to its root or base form (e.g., 'running' to 'run'), often using simple suffix stripping?",
      "options": ["Stemming", "Named Entity Recognition", "Lemmatization", "Syntactic Analysis"],
      "correct_answer": "Stemming"
    },
    {
      "id": 43,
      "question": "Which specific technique is used to remove common, high-frequency words that carry little semantic meaning (like 'the', 'a', 'is') from a text corpus?",
      "options": ["Part-of-Speech Tagging", "Stop Word Removal", "Chunking", "Bag-of-Words Model"],
      "correct_answer": "Stop Word Removal"
    },
    {
      "id": 44,
      "question": "What is the goal of **Named Entity Recognition (NER)**?",
      "options": ["To classify a document by topic.", "To identify and categorize proper nouns (like people, places, organizations).", "To determine the sentiment (positive/negative) of a text.", "To convert text into a vector space model."],
      "correct_answer": "To identify and categorize proper nouns (like people, places, organizations)."
    },
    {
      "id": 45,
      "question": "Which model represents text as an unordered collection of words, disregarding grammar and word order, only keeping track of word frequency?",
      "options": ["Transformer Model", "N-gram Model", "Recurrent Neural Network", "Bag-of-Words Model"],
      "correct_answer": "Bag-of-Words Model"
    },
    {
      "id": 46,
      "question": "What is the name of the statistical measure used to evaluate the importance of a word in a document relative to a corpus, based on term frequency and inverse document frequency?",
      "options": ["Word2Vec", "TF-IDF (Term Frequency-Inverse Document Frequency)", "Cosine Similarity", "Perplexity"],
      "correct_answer": "TF-IDF (Term Frequency-Inverse Document Frequency)"
    },
    {
      "id": 47,
      "question": "Which architecture introduced the concept of **attention mechanisms** to weigh the importance of different words in a sequence, revolutionizing sequence-to-sequence tasks like machine translation?",
      "options": ["Long Short-Term Memory (LSTM)", "Convolutional Neural Network (CNN)", "Recurrent Neural Network (RNN)", "Transformer"],
      "correct_answer": "Transformer"
    },
    {
      "id": 48,
      "question": "The task of determining the intended meaning of an ambiguous word based on its context (e.g., 'bank' as a financial institution or river edge) is known as:",
      "options": ["Syntactic Parsing", "Text Summarization", "Word Sense Disambiguation (WSD)", "Coreference Resolution"],
      "correct_answer": "Word Sense Disambiguation (WSD)"
    },
    {
      "id": 49,
      "question": "What is a **Word Embedding**?",
      "options": ["A text file containing common English words.", "A matrix that stores the results of sentiment analysis.", "A dense, low-dimensional vector representation of a word's meaning.", "A set of rules for grammatical correctness."],
      "correct_answer": "A dense, low-dimensional vector representation of a word's meaning."
    },
    {
      "id": 50,
      "question": "Which pre-trained language model, developed by Google, is famous for using a **bidirectional** training approach to learn context from both the left and right sides of a word simultaneously?",
      "options": ["GPT-3", "BERT (Bidirectional Encoder Representations from Transformers)", "Word2Vec", "ELMo (Embeddings from Language Models)"],
      "correct_answer": "BERT (Bidirectional Encoder Representations from Transformers)"
    },
    {
      "id": 51,
      "question": "In the context of evaluating a language model, what does **Perplexity** measure?",
      "options": ["The model's speed in generating text.", "How well the model can generalize to unseen vocabulary.", "How confident the model is in its own predictions (lower is better).", "The model's ability to handle multiple languages."],
      "correct_answer": "How confident the model is in its own predictions (lower is better)."
    },
    {
      "id": 52,
      "question": "The task of automatically generating a concise and coherent summary of a longer text is called:",
      "options": ["Machine Translation", "Topic Modeling", "Text Summarization", "Question Answering"],
      "correct_answer": "Text Summarization"
    },
    {
      "id": 53,
      "question": "Which sub-task of NLP is concerned with analyzing the grammatical structure of sentences to define relationships between words (e.g., Subject, Object)?",
      "options": ["Lexical Analysis", "Sentiment Analysis", "Syntactic Parsing", "Pragmatic Analysis"],
      "correct_answer": "Syntactic Parsing"
    },
    {
      "id": 54,
      "question": "The **BLEU Score** is a common metric used in NLP to evaluate the quality of which specific task?",
      "options": ["Part-of-Speech Tagging", "Text Classification", "Machine Translation", "Named Entity Recognition"],
      "correct_answer": "Machine Translation"
    },
    {
      "id": 55,
      "question": "Which recurrent architecture was introduced specifically to solve the vanishing gradient problem in standard RNNs by using gates (Input, Forget, Output)?",
      "options": ["Transformer", "Perceptron", "LSTM (Long Short-Term Memory)", "Simple RNN"],
      "correct_answer": "LSTM (Long Short-Term Memory)"
    },
    {
      "id": 56,
      "question": "What is **Coreference Resolution**?",
      "options": ["Identifying the primary subject of a paragraph.", "Determining if two documents are about the same topic.", "Finding all expressions that refer to the same entity in a text.", "Analyzing the emotional tone of a sentence."],
      "correct_answer": "Finding all expressions that refer to the same entity in a text."
    },
    {
      "id": 57,
      "question": "Which classification task involves identifying and extracting subjective information from source materials, typically categorizing opinions as positive, negative, or neutral?",
      "options": ["Topic Modeling", "Information Retrieval", "Text Generation", "Sentiment Analysis"],
      "correct_answer": "Sentiment Analysis"
    },
    {
      "id": 58,
      "question": "In the context of language model training, what is **Fine-Tuning**?",
      "options": ["Training a model from scratch with a small dataset.", "Using a pre-trained model and continuing to train it on a smaller, task-specific dataset.", "Adjusting the model's hyper-parameters without changing the weights.", "Removing unnecessary layers from a deep neural network."],
      "correct_answer": "Using a pre-trained model and continuing to train it on a smaller, task-specific dataset."
    },
    {
      "id": 59,
      "question": "What kind of NLP model is typically used for **Topic Modeling**, where the goal is to discover abstract 'topics' that occur in a collection of documents?",
      "options": ["Sequence-to-Sequence (Seq2Seq)", "Latent Dirichlet Allocation (LDA)", "Conditional Random Field (CRF)", "Support Vector Machine (SVM)"],
      "correct_answer": "Latent Dirichlet Allocation (LDA)"
    },
    {
      "id": 60,
      "question": "In the process of **Tokenization**, what term refers to combining frequent sequences of characters into a single token, which helps handle out-of-vocabulary words in large models like BERT and GPT?",
      "options": ["WordPiece (Subword Tokenization)", "Chunking", "Phoneme Mapping", "Lexical Simplification"],
      "correct_answer": "WordPiece (Subword Tokenization)"
    }
]


ATTEMPTS_DATA = [
    {
        "user_id": 1,
        "quiz_id": 1,
        "total_scored": 100
    },
    {
        "user_id": 2,
        "quiz_id": 1,
        "total_scored": 60
    },
    {
        "user_id": 3,
        "quiz_id": 1,
        "total_scored": 80
    },
]

def seed_data():
    # Add users
    for user in USERS_DATA:
        new_user = User(
            id=user['id'],
            username=user['username'],
            fullname=user['fullname'],
            is_admin=user['is_admin']
        )
        new_user.set_password(user['password'])
        db.session.add(new_user)
    db.session.commit()

    # Add subjects
    for sub in SUBJECT_DATA:
        new_subject = Subject(
            id=sub['id'],
            name=sub['name'],
            description=sub['description']
        ) 
        db.session.add(new_subject)
    db.session.commit()

    # Add quiz
    for qq in QUIZ_DATA:
        new_quiz = Quiz(
            id=qq['id'],
            name=qq['name'],
            description=qq['description'],
            date_of_quiz=qq['date_of_quiz'],
            duration=qq['duration'],
            subject_id=qq['subject_id']
        )
        db.session.add(new_quiz)
    db.session.commit()

    for q in QUIZ_COMPUTER_VISION_QUESTIONS:
        quiz_id = 1
        new_question = QuizQuestion(
            id=q['id'],
            quiz_id=quiz_id,
            question_statement=q['question'],
            option_1=q['options'][0],
            option_2=q['options'][1],
            option_3=q['options'][2],
            option_4=q['options'][3],
            correct_option=q['correct_answer']
        )
        db.session.add(new_question)
    db.session.commit()

    for q in QUIZ_DATA_AI_PYTHON_QUESTIONS:
        quiz_id = 2
        new_question = QuizQuestion(
            id=q['id'],
            quiz_id=quiz_id,
            question_statement=q['question'],
            option_1=q['options'][0],
            option_2=q['options'][1],
            option_3=q['options'][2],
            option_4=q['options'][3],
            correct_option=q['correct_answer']
        )
        db.session.add(new_question)
    db.session.commit()

    for q in QUIZ_DATA_NLP:
        quiz_id = 3
        new_question = QuizQuestion(
            id=q['id'],
            quiz_id=quiz_id,
            question_statement=q['question'],
            option_1=q['options'][0],
            option_2=q['options'][1],
            option_3=q['options'][2],
            option_4=q['options'][3],
            correct_option=q['correct_answer']
        )
        db.session.add(new_question)
    db.session.commit()

    for a in ATTEMPTS_DATA:
        new_attempt = QuizScore(
            user_id=a['user_id'],
            quiz_id=a['quiz_id'],
            total_scored=a['total_scored']
        )
        db.session.add(new_attempt)
    db.session.commit()

