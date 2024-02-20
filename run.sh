# MoNuSeg
python ./MTGFFNet_train.py --ds monuseg --device cuda --config3 MTGFFMonuSeg # 0.640
python ./HoverNet_train.py --ds monuseg --device cuda --config HoverNetMoNuSeg # PQ 0.604
python ./CDNet_train.py --ds monuseg --device cuda --config CDNet_MoNuSeg # PQ 0.56
#python ./CenterMask.py --ds monuseg --device cuda --config HoverConfig1
python ./MaskRCNN_train.py --ds monuseg --device cuda --config MrcnnMonuSeg # PQ 0.544

# dsb
python ./MTGFFNet_train.py --ds dsb --device cuda --config MTGFFDSB # PQ 0.72
python ./HoverNet_train.py --ds dsb --device cuda --config HoverNetDSB # PQ 0.710686
python ./CDNet_train.py --ds dsb --device cuda --config CDNet_DSB 
python ./MaskRCNN_train.py --ds dsb --device cuda --config MrcnnDSB 

# cpm
python ./MTGFFNet_train.py --ds cpm --device cuda --config MTGFFCPM # PQ 0.65
python ./HoverNet_train.py --ds cpm --device cuda --config HoverNetCPM # 0.64
python ./CDNet_train.py --ds cpm --device cuda --config CDNet_CPM # 0.590
python ./MaskRCNN_train.py --ds cpm --device cuda --config MrcnnCPM # 0.609

# pannuke
python ./MTGFFNet_train.py --ds pannuke --device cuda --config MTGFFPanNuke #0.60
python ./HoverNet_train.py --ds pannuke --device cuda --config HoverPannuke 
python ./CDNet_train.py --ds pannuke --device cuda --config CDNet_P
python ./MaskRCNN_train.py --ds pannuke --device cuda --config MrcnnMonuSeg

# conic
python ./MTGFFNet_train.py --ds conic --device cuda --config MTGFFConic
python ./HoverNet_train.py --ds conic --device cuda --config HoverNetConic
python ./CDNet_train.py --ds conic --device cuda --config CDNet_Conic
python ./MaskRCNN_train.py --ds conic --device cuda --config MrcnnConic
