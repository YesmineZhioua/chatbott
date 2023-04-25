from typing import List
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorboard import data
from keras import Sequential
from keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")
import classes as classes
import nltk #NLTK est une bibliothèque Python utilisée pour le traitement automatique du langage naturel (NLP)

# initialisation de lemmatizer pour obtenir la racine des mots
lemmatizer = WordNetLemmatizer()


with open("intents.json") as file:
  data = json.load(file)
# création des listes
words = []
classes = []
doc_X = []
doc_y = []
# parcourir avec une boucle For toutes les intentions
# tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
# le tag associé à l'intention sont ajoutés aux listes correspondantes
for intent in data["intents"]:
  for pattern in intent["patterns"]:
    tokens = nltk.word_tokenize(pattern)
    words.extend(tokens)
    doc_X.append(pattern)
    doc_y.append(intent["tag"])

  # ajouter le tag aux classes s'il n'est pas déjà là
  if intent["tag"] not in classes:
    classes.append(intent["tag"])
# lemmatiser tous les mots du vocabulaire et les convertir en minuscule
# si les mots n'apparaissent pas dans la ponctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# trier le vocabulaire et les classes par ordre alphabétique et prendre le
# set pour s'assurer qu'il n'y a pas de doublons
words = sorted(set(words))
classes = sorted(set(classes))
# liste pour les données d'entraînement
training = []
out_empty = [0] * len(classes)
# création du modèle d'ensemble de mots
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # marque l'index de la classe à laquelle le pattern atguel est associé à
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # ajoute le one hot encoded BoW et les classes associées à la liste training
    training.append([bow, output_row])
# mélanger les données et les convertir en array
random.shuffle(training)
training = np.array(training, dtype=object)
# séparer les features et les labels target
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))
# définition de quelques paramètres
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200
# modèle Deep Learning
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
# entraînement du modèle
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

def clean_text(text):
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens
def bag_of_words(text, vocab):
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens:
    for idx, word in enumerate(vocab):
      if word == w:
        bow[idx] = 1
  return np.array(bow)
def pred_class(text, vocab, labels):
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list
def get_response(intents_list, intents_json):
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents:
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result


# lancement du chatbot
print("start talking with the bot(type quit to stop)!")
while True:
  inp = input("You : ")
  if inp.lower() == "quit":
    break
  while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)
























######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!################
# . Elle offre des fonctionnalités telles que la tokenisation, la lemmatisation, l'analyse syntaxique, la reconnaissance d'entités nommées et la classification de texte.
# import numpy as np #manipulation des tableaux
# from nltk import WordNetLemmatizer
# from nltk.stem.lancaster import LancasterStemmer
# stemmer = LancasterStemmer()
# #la racinisation des mots d'un texte. La racinisation est le processus de réduction d'un mot à sa racine ou à sa forme de base
# import numpy
# import tflearn #facilite la création et la formation de réseaux de neurones pour l'apprentissage automatique.
# # Elle fournit des fonctions pour créer rapidement des réseaux de neurones et entraîner des modèles à partir de données
# import tensorflow #utilisée pour des tâches d'IA telles que la reconnaissance d'images et la classification de texte, traduction automatique.
# import random
# import json
# import curses
# import pickle #Pickle est un module de Python qui permet de sauvegarder des objets Python complexes sous forme binaire et de les restaurer plus tard.

#
# with open("intents.json") as file:
#   data = json.load(file)
#
# try:
#   with open("data.pickle", "rb") as f:
#    words, labels, training, output = pickle.load(f)
#
# except:
# ##1) séparation des données
# #création des listes
#   words = []
#   labels = []
#   docs_x = []
#   docs_y = []
#   for intent in data["intents"]: #parcourir toutes les intentions
#     for pattern in intent["patterns"]:
#       wrds = nltk.wordpunct_tokenize(pattern) #tokeniser chaque pattern
#       words.extend(wrds) #ajouter les tokens à la  liste words
#       docs_x.append(pattern) #trajaa lista feha les questions kol(les patterns)
#       docs_y.append(intent["tag"])# liste feha les classes ta kol question f lista docs_x(les tags correspondants)
#       # ajouter le tag aux classes s'il n'est pas déjà là
#       if intent["tag"] not in labels:
#         labels.append(intent["tag"])
#   #la liste "words" est transformée en une liste de mots normalisés (stems) en minuscules, à l'exception des points d'interrogation.
#   #convertir chaque mot en minuscules avant la normalisation
#   words = [stemmer.stem(w.lower()) for w in words if w != "?"]
#   # trier le vocabulaire et les classes par ordre alphabétique et prendre le
#   # set pour s'assurer qu'il n'y a pas de doublons
#   words = sorted(list(set(words)))
#   labels = sorted(labels)
# ##1) Traitement des données
# # liste pour les données d'entraînement
#   training = []
#   out_empty = [0 for _ in range(len(labels))] # = [0] * len(classes)
# #création du modéle d'ensemble de mots
#   for x, doc in enumerate(docs_x):
#      bag = []
#      wrds = [stemmer.stem(w) for w in doc]
#      for w in words:
#         if w in wrds:
#            bag.append(1)
#         else:
#            bag.append(0)
# #marquer l'index de la clasee à laquell le pattern actuel est associé à
#      output_row = out_empty[:]
#      output_row[labels.index(docs_y[x])] = 1
#      # ajoute le one hot encoded BoW et les classes associées à la liste training
#      training.append(bag)
#      output.append(output_row)
#
# #convertir les données en array
#   training = numpy.array(training)
#   output = np.array(output)
#   with open("data.pickle", "wb") as f:
#     pickle.dump((words, labels, training, output), f)
# #nbr de ligne inconnue(none) , nbr de colonnes = la longueur des données d'entrée d'entraînement.
# net = tflearn.input_data(shape=[None, len(training[0])])
# #deux autre couche entièrement connectée au réseau avec 8 neurones.
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# # connectée finale au réseau avec un nombre de neurones égal à la longueur des données de sortie.
# # Le paramètre activation spécifie la fonction d'activation à utiliser pour la couche de sortie.
# # fonction softmax qui est couramment utilisée pour les problèmes de classification multi-classes.
# #La fonction d'activation softmax est utilisée pour la couche de sortie pour garantir que les sorties représentent une distribution de probabilité sur les classes possibles.
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# #crée une couche de régression qui est utilisée pour entraîner le réseau de neurones.
# net = tflearn.regression(net)
#
# model = tflearn.DNN(net) #utilisé pour entraîner et évaluer le modèle de réseau de neurones.
# try:
#   model.load("model.tflearn")
# except:
#   #training : les données d'entrée d'entraînement.
# #output : les données de sortie attendues pour chaque entrée d'entraînement.
# #n_epoch : le nombre d'époques (itérations) pour entraîner le modèle.
# #batch_size : la taille de chaque lot (batch) de données d'entraînement utilisé pour mettre à jour les poids du modèle pendant l'entraînement. Une taille de lot plus grande accélère l'entraînement, mais peut nécessiter plus de mémoire.
#   #cette ligne de code entraîne le modèle de réseau de neurones avec les données d'entraînement fournies pendant 1000 époques, en utilisant des lots de 8 exemples à la fois.
#   # Les mesures de précision et de perte seront affichées pendant l'entraînement pour surveiller la progression du modèle
#   model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#   model.save("model.tflearn")
# def bag_of_words(s, words):
#     bag = [0 for _ in range(len(words))]
#     s_words = nltk.word_tokenize(s)
#     s_words = [stemmer.stem(word.lower()) for word in s_words]
#
#     for se in s_words:
#       for i, w in enumerate(words):
#         if w == se:
#           bag[i] =1
#     return numpy.array(bag)
# def chat():
#   print("start talking with the bot(type quit to stop)!")
#   while True:
#     inp = input("You : ")
#     if inp.lower() == "quit":
#       break
#     #: Cette ligne de code prédit la classe de la nouvelle entrée "inp" en la convertissant en sac de mots et en utilisant le modèle entraîné pour prédire la classe associée.
#     # Le résultat est un tableau contenant les scores de probabilité pour chaque classe possible.
#     results = model.predict([bag_of_words(inp, words)])[0]
#     #Cette ligne de code trouve l'index de la classe avec la plus haute probabilité dans le tableau "results".
#     # numpy.argmax renvoie l'indice de la première occurrence du plus grand élément dans le tableau.
#     results_index = numpy.argmax(results)
#     # Cette ligne de code associe la classe prédite "tag" avec l'étiquette "labels" correspondante à l'aide de l'indice "results_index".
#     # La variable "labels" contient les noms des classes de l'ensemble de données d'entraînement utilisées pour former le modèle.
#     ## ==> le code prédit la classe de la nouvelle entrée "inp" en utilisant le modèle entraîné et en renvoie l'étiquette associée à la classe prédite.
#     tag = labels[results_index]
#     print(results)
#   #  if results[results_index] > 0.3
#     for tg in data["intents"]:
#      if tg['tag'] == tag:
#       responses = tg['responses']
#
#     print(random.choice(responses))
#    # else:
#      # print("I didn't get that , try again .")
#
# chat()




