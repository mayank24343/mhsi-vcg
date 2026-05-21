# chair evall on dataset
python chair_eval_main.py  --impl 1  --annotation_file "C:\Users\Mayank\fiftyone\coco-2017\validation\labels.json"   --image_dir "C:\Users\Mayank\fiftyone\coco-2017\validation\data" --alphas 0 0.1 0.5 1   --num_images 50  --save_dir eval_results  
# pope eval on dataset 
python pope_eval_main.py  --impl 1  --annotation_file "C:\Users\Mayank\fiftyone\coco-2017\validation\labels.json"   --image_dir "C:\Users\Mayank\fiftyone\coco-2017\validation\data" --alphas 0 -0.1 -0.5 -1 0.1 0.5 1   --num_images 50    --questions_per_image 6    --save_dir eval_results 

# individual run
python main.py --image "C:\Users\Mayank\OneDrive\Desktop\MHSI\Screenshot 2026-05-20 102949.png" --question "Provide a detailed description of the image?" --impl 1   