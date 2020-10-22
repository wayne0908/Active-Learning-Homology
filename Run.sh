for i in {1..15..1}
do
	python main.py --W=0.1 --Tau=5 --Graph='Radius' --Comp='VR' --S=0 --L=1.5 --LoadRipser=0 --LoadData=0 --DrawDist=1 --DrawPD=1 --DrawBoundary=1 --Trial=$i
	python main.py --W=0.3 --Tau=5 --Graph='Radius' --Comp='VR' --S=0 --L=1.5 --LoadRipser=0 --LoadData=0 --DrawDist=1 --DrawPD=1 --DrawBoundary=1 --Trial=$i
	python main.py --W=0.5 --Tau=5 --Graph='Radius' --Comp='VR' --S=0 --L=1.5 --LoadRipser=0 --LoadData=0 --DrawDist=1 --DrawPD=1 --DrawBoundary=1 --Trial=$i
done


# python main.py --W=0.1 --Graph='Radius' 
# python main.py --W=0.3 --Graph='Radius' 
# python main.py --W=0.5 --Graph='Radius' 
# python main.py --W=0.1 --Graph='NN' --Comp='LVR' --S=5 --Load=$false
# python main.py --W=0.3 --Graph='NN' --Comp='LVR' --S=5 --Load=$false
# python main.py --W=0.5 --Graph='NN' --Comp='LVR' --S=5 --Load=$false

