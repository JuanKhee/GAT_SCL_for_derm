from models.CNN import CNNModel
from models.GAT_superpixel import GAT_image

from SkinDiseaseClassifierCNN import SkinDiseaseClassifier as CNNClassifier
from SkinDiseaseClassifierSCL import SkinDiseaseClassifier as CNNSCLClassifier
from SkinDiseaseClassifierMTSCL import SkinDiseaseClassifier as CNNMTSCLClassifier
from SkinDiseaseClassifierGAT import SkinDiseaseClassifier as GATClassifier
from SkinDiseaseClassifierGAT_SCL import SkinDiseaseClassifier as GATSCLClassifier
from SkinDiseaseClassifierGAT_MTSCL import SkinDiseaseClassifier as GATMTSCLClassifier
from SkinDiseaseClassifierGATCNN import SkinDiseaseClassifier as GATCNNClassifier
# from SkinDiseaseClassifierCNN import SkinDiseaseClassifier as CNNClassifier
# from SkinDiseaseClassifierCNN import SkinDiseaseClassifier as CNNClassifier

def load_classifier(key):
    if key == 'CNN':
        return CNNClassifier
    if key == 'CNNSCL':
        return CNNSCLClassifier
    if key == 'CNNMTSCL':
        return CNNMTSCLClassifier
    if key == 'GAT':
        return GATClassifier
    if key == 'GATSCL':
        return GATSCLClassifier
    if key == 'GATMTSCL':
        return GATMTSCLClassifier



def run_model(classifier_key, hyperparameter):
    SkinDiseaseClassifier = load_classifier(key)
    epochs = learning_rate
    dev_classifier = SkinDiseaseClassifier(vgg16_model, epochs=2, output_dir='dev_model_result')
    dev_classifier.create_dataloader(
        train_root_path='dev_images/train',
        test_root_path='dev_images/test',
        seed=57
    )
    dev_classifier.train_model()
    dev_classifier.load_model()
    dev_classifier.evaluate_model()

