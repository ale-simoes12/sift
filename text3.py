import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

# Diretórios das imagens e de saída
image_dir = 'C:/Users/aless/OneDrive/Documentos/sift/Etapa2/imgs/'
image_out = 'C:/Users/aless/OneDrive/Documentos/sift/Etapa2/imgswd/'

# Listar todos os arquivos de imagem no diretório
image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

print("image files:", image_files)

# Inicializar dicionários e lista
vector = {}
imgs_des = {}
imgs_kp = {}
lista = []
lista_images =[]

# Processar cada imagem
for file in image_files:
    img = cv.imread(file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Criar o detector SIFT
    sift = cv.SIFT_create()
    
    # Detectar keypoints e calcular descritores
    kp, des = sift.detectAndCompute(gray, None)
    
    # Ordenar keypoints com base na resposta e selecionar os melhores
    #keypoints = sorted(kp, key=lambda x: x.response, reverse=True)
    #num_best_keypoints = 4000
    #keypoints = keypoints[:num_best_keypoints]
    lista_images.append(gray)
    # Desenhar os keypoints na imagem
    img_with_keypoints = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output_file = os.path.splitext(file)[0] + '_keypoints.jpg'
    novo_path = output_file.replace("imgs", "imgswd")
    cv.imwrite(novo_path, img_with_keypoints)
    lista.append(novo_path)
    imgs_des[novo_path] = des
    imgs_kp[novo_path] = kp

print(imgs_des[lista[0]])
print("==============================================================")
print(imgs_des[lista[1]])
print("==============================================================")
print(imgs_kp[lista[0]])
print("==============================================================")
print(imgs_kp[lista[1]])
print("==============================================================")
print(lista_images[0])
print(lista_images[1])
# bf = cv.BFMatcher()
# matches = bf.knnMatch(imgs_des[lista[0]],imgs_des[lista[1]],k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(lista_images[0],imgs_kp[lista[0]],lista_images[1],imgs_kp[lista[1]],good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()


# # print(imgs_des[lista[0]].dtype, imgs_des[lista[1]].dtype)
# # # create BFMatcher object

# # # create BFMatcher object
# # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.knnMatch(imgs_des[lista[0]],imgs_des[lista[1]],k=2)

# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv.drawMatchesKnn(lista_images[0],imgs_kp[lista[0]],lista_images[1],imgs_kp[lista[1]],matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()




# Cria o objeto BFMatcher para encontrar correspondências entre os descritores
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# Encontra as correspondências entre os descritores das duas imagens
matches = bf.match(imgs_des[lista[0]], imgs_des[lista[1]])

# Ordena as correspondências pela distância (quanto menor, melhor)
matches = sorted(matches, key=lambda x: x.distance)


print("metches" , matches)

# Controle o número de correspondências a serem desenhadas
numero_de_matches = 100  # Altere este valor para controlar quantas correspondências deseja exibir


# Função para verificar se duas linhas se cruzam
def linhas_se_cruzam(p1, p2, q1, q2):
    def orientacao(a, b, c):
        return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    
    o1 = orientacao(p1, p2, q1)
    o2 = orientacao(p1, p2, q2)
    o3 = orientacao(q1, q2, p1)
    o4 = orientacao(q1, q2, p2)
    
    return o1 * o2 < 0 and o3 * o4 < 0

# Filtrar correspondências que não cruzam outras linhas
matches_filtrados = []

for i in range(len(matches)):

    pt1_1 = imgs_kp[lista[0]][matches[i].queryIdx].pt
    pt2_1 = (imgs_kp[lista[1]][matches[i].trainIdx].pt[0] + lista_images[0].shape[1], imgs_kp[lista[1]][matches[i].trainIdx].pt[1])
    
    cruza = False
    for j in range(len(matches_filtrados)):
        pt1_2 = imgs_kp[lista[0]][matches_filtrados[j].queryIdx].pt
        pt2_2 = (imgs_kp[lista[1]][matches_filtrados[j].trainIdx].pt[0] + lista_images[0].shape[1], imgs_kp[lista[1]][matches_filtrados[j].trainIdx].pt[1])
        
        if linhas_se_cruzam(pt1_1, pt2_1, pt1_2, pt2_2):
            cruza = True
            break
    
    if not cruza:
        matches_filtrados.append(matches[i])
    
    if len(matches_filtrados) >= numero_de_matches:
        break




# Cria uma imagem para exibir as correspondências lado a lado
imagem_matches = cv.hconcat([lista_images[0], lista_images[1]])



# Desenha manualmente as correspondências com linhas mais grossas
for match in matches_filtrados:

    pt1 = tuple(map(int, imgs_kp[lista[0]][match.queryIdx].pt))
    pt2 = (int(imgs_kp[lista[1]][match.trainIdx].pt[0] + lista_images[0].shape[1]), int(imgs_kp[lista[1]][match.trainIdx].pt[1]))
    cv.line(imagem_matches, pt1, pt2, (0, 0, 255), 5)  # Cor verde e espessura 3

print(matches_filtrados)

  # Mostra quais pontos correspondem entre si
for i, match in enumerate(matches[:100]):  # Ajuste o número conforme necessário
    pt1 = imgs_kp[lista[0]][match.queryIdx].pt
    pt2 = imgs_kp[lista[1]][match.trainIdx].pt
    print(f'Match {i+1}: Imagem 1 ponto {pt1} -> Imagem 2 ponto {pt2}')

for i, match in enumerate(matches[:10]):  # Ajuste o número conforme necessário
    pt1 = imgs_kp[lista[0]][match.queryIdx].pt
    pt2 = imgs_kp[lista[1]][match.trainIdx].pt
    des1 = imgs_des[lista[0]][match.queryIdx]
    

    des2 = imgs_des[lista[1]][match.trainIdx]
    print(f'Match {i+1}:')
    print(f'  Imagem 1 ponto: {pt1}, descritor: {des1}')
    print(f'  Imagem 2 ponto: {pt2}, descritor: {des2}')    

#Exibe a imagem com as correspondências
plt.figure(figsize=(15, 10))
plt.imshow(cv.cvtColor(imagem_matches, cv.COLOR_BGR2RGB))
plt.title(f'Correspondências entre as duas perspectivas usando SIFT ({len(matches_filtrados)} matches)')
plt.axis('off')
plt.show()























# FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(imgs_des[lista[0]],imgs_des[lista[1]],k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)

# img3 = cv.drawMatchesKnn(lista_images[0],imgs_kp[lista[0]],lista_images[1],imgs_kp[lista[1]],matches,None,**draw_params)
# plt.imshow(img3,),plt.show()