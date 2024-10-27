from task4 import *
import argparse
import sys
import os
import csv

def generate_part1_csv(image_dir, output_file_name="output_csv"):
    output_csv_path = os.path.join(os.getcwd(), output_file_name)
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Image Name', 'Number of sutures', 'Mean Inter Suture Spacing', 'Variance of Inter Suture Spacing', 'Mean Suture Angle wrt X-axis', 'Variance of Suture Angle wrt X-axis'])
        
    image_file_name_list = os.listdir(image_dir)
    for image_file_name in image_file_name_list:
        image_path = os.path.join(image_dir, image_file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        suture_count, mean_distance, var_distance, angle_mean, angle_var = extract_image_features(image)
        
        with open(output_csv_path, "a") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([image_file_name, suture_count, mean_distance, var_distance, angle_mean, angle_var])


def generate_part2_csv(image_dir, input_csv_path, output_csv_file_name="output_csv2"):
    # input_csv_path = os.path.join(os.getcwd(), input_csv_file_name)
    output_csv_path = os.path.join(os.getcwd(), output_csv_file_name)
    if not os.path.exists(output_csv_path):
            with open(output_csv_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['img1_name', 'img2_name', 'output_distance', 'output_angle'])

    with open(input_csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            img1_name = row['img1_name']
            img2_name = row['img2_name']
            
            img1_path = os.path.join(image_dir, img1_name)
            img2_path = os.path.join(image_dir, img2_name)

            image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            _, _, var_distance1, _, angle_var1 = extract_image_features(image1)
            _, _, var_distance2, _, angle_var2 = extract_image_features(image2)

            if var_distance1<var_distance2:
                output_distance = 1
            else:
                output_distance = 2

            if angle_var1<angle_var1:
                output_angle = 1
            else:
                output_angle = 2

            with open(output_csv_path, "a") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([img1_name, img2_name, output_distance, output_angle])
    


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python3 main.py <task> <img dir or input csv> <output csv>")
        sys.exit(1)

    task = sys.argv[1]
    input_path = sys.argv[2]
    output_csv = sys.argv[3]
    
    if task == '1':
        image_dir = input_path
        generate_part1_csv(image_dir=image_dir, output_file_name=output_csv)
        print(f"CSV file successfully generated at {output_csv}.")
    elif task == '2':
        print(input_path)
        generate_part2_csv(image_dir="data", input_csv_path=input_path, output_csv_file_name=output_csv)
        print(f"Comparison results written to {output_csv}.")
    else:
        print("Invalid task. Please use '1' or '2' for the task.")
        sys.exit(1)


