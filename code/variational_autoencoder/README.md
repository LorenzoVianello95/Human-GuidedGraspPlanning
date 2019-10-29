save_depth_color.py = save the dataset of color and depth images to be passed to dexnet. 

(build_masks.py = for each color image in data/rgbd create a mask using man labelling.)

Create grasps dataset= using Dexnet for each image collect all the possible grasps

var_autoenc.py = crea encoder and decoder model

weight 000..00235 buono, fatto con conv+ var latent layer 676 , unico problema e' che depth condizionata da rgb( no peso extra ) associato a var_enc_conv

var_enc_conv_split dovrebbe essere un tentativo di considerare in maniera indipendente depth e rgb prima parte di convolution viene fatta separatamente


##DEnse plus attempt with variational

RIASSUMENDO PIU O MENO QUELLO CHE E' STATO FATTO CON I RELATIVI RISULTATI:

IN UN PRIMO MOMENTO HO PROVATO A TRATTARE LE INPUT IMAGES AS VECTOR DI DIMENSIONE 4*100*100 E DOPO UNA PARTE DI PREPROCESSING TRAMITE DEI DENSE LAYERS 
PASSARLA AL VARIATIONAL LEVEL, IL RISULTATO PERO NON E' STATO SODDISFACENTE. oltretutto fare dense e' molto dispendioso in fatto di tempisticche.

![Alt text](variational_autoencoder/pict/vae_mlp_encoder21597.png?raw=true "Dense and variational")

link a cui faccio riferimento per variational layers

https://keras.io/examples/variational_autoencoder/
https://becominghuman.ai/variational-autoencoders-for-new-fruits-with-keras-and-pytorch-6d0cfc4eeabd
https://blog.keras.io/building-autoencoders-in-keras.html


##CONVOLUTIONAL
 
HO QUINDI PROVATO A PASSARE LA MATRICE 4,100,100 DENTRO A UNA CONVOLUTIONAL NETWORK, IN QUESTO CASO IL RISULTATO E' STATO PIU SODDISFACENTE MA HO RISCONTRATO ALCUNI PROBLEMI:
1- LA DIMENSIONE DEL LATENT LAYER AUMENTA NOTEVOLMENTE
2- ANCHE IN QUESTO CASO IL FATTO DI AGGIUNGERE IL VARIATIONAL NON MIGLIORA
3- PROCESSANDO INSIEME DEPTH E RGB LA DEPTH IMAGE VIENE A DIPENDERE DA RGB CREANDO DELLE ANOMALIE NON VOLUTE... SI VEDA IMMAGINE

![Alt text](variational_autoencoder/pict/vae_mlp_encoder.png?raw=true "Dense and variational")

![Alt text](variational_autoencoder/pict/Figure_prov2.png?raw=true "Title")
![Alt text](variational_autoencoder/pict/Figure_prov3.png?raw=true "Title")
![Alt text](variational_autoencoder/pict/Figure_prov4.png?raw=true "Title")
![Alt text](variational_autoencoder/pict/Figure_prov5.png?raw=true "Title")


##convolutional split
PARTENDO DAL PRESUPPOSTO DEL PUNTO 3 E TENENDO A MENTE I CONSIGLI DI QUESTO PAPER https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8715446 
HO PROCESSATO LA DEPTHE E L RGB IN DUE CONVOLUTIONAL INDIPENDENT CHANNELS,
COSI FACENDO E DANDO UN ADEGUATO PESO ALLA DEPTH RISPETTO AL RGB HO OTTENUTO BUONI RISULTATI,
IN QUESTO CASO LA DIMENSIONE DEL LATENT LAYER E' 1000. La loss per come lho definita L= w_d*(diff_im_depth)+w_rgb(diff_img_rgb) raggiunge valori minimi con del data augmentation, in particolare aggiungo al dataset la trsposta e il fliprl.
Valore minimimo ragiunto L= 0.0210 forse si puo' scendere ancora tramite altro data augmentation...

Se vuoi usare un modello gia allenato "python autoenc_conv_split.py -w weights00000480loss231.h5 oppure 075loss.. " 
 
HO FATTO ANCHE IMPLEMENTAZIONE CON CONV+SPLIT+VAR e funziona abbastanza bene, si assesta intorno ai 0.0345 l'apprendimento e' piu lento e costante rispetto a solo convolutional mi pare, ora vedo quale e' di fatto la soluzione migliore, c'e la vaga possibilita' che lasciando allenare ancora possa migliorare, ma non lo penso molto possibile, comunque una cosa che ho notato e' che devo dare un bassissimo peso al kl loss altrimenti si assesta, non so bene come gestire sta cosa. ho fatto anche una mezza prova senza kl loss e sembra funzionare bene... FUCK...
In generale pero meglio senza...

esempio autoencoder cifar10 con latent layer 750 circa https://github.com/jellycsc/PyTorch-CIFAR-10-autoencoder
							http://users.cecs.anu.edu.au/~Tom.Gedeon/conf/ABCs2018/paper/ABCs2018_paper_65.pdf (meglio ancora forse)

![Alt text](variational_autoencoder/pict/conv_split/vae_mlp_encoder.png?raw=true "Encoder")
![Alt text](variational_autoencoder/pict/conv_split/vae_mlp_decoder.png?raw=true "Decoder")
![Alt text](variational_autoencoder/pict/conv_split/data_augm231loss.png?raw=true "Title")
![Alt text](variational_autoencoder/pict/conv_split/Figure_1-1.png?raw=true "Title")
![Alt text](variational_autoencoder/pict/conv_split/Figure_1-2.png?raw=true "Title")


Training in generale e' tanto piu veloce quanti piu convolutional metto e meno dense questi degli esempi, si vede che si ha overfitting molto velocemente nel primo caso:

![Alt text](variational_autoencoder/pict/graph2.png?raw=true "Title")
![Alt text](variational_autoencoder/pict/var_conv_plot.png?raw=true "Title")


