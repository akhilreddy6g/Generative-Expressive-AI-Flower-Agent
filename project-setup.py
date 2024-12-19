import bpy
import pickle
import subprocess
import sys
import sklearn
import warnings
import numpy as np
from transformers import pipeline

def set_rose_to_default_position():
    # Get the armature by name
    armature = bpy.data.objects.get("Petals-Movement")
    # Check if the armature exists and is indeed an armature
    if armature and armature.type == 'ARMATURE':
        # Set the armature as the active object and select it
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        # Switch to Pose Mode
        bpy.ops.object.mode_set(mode='POSE')
        # Get the pose bones
        pose_bones = armature.pose.bones
        # Iterate through the pose bones
        for bone in pose_bones:
            if bone.name == "Onc-Controller":
                # Set location coordinates
                bone.location = (0, 0, 0)
                # Set rotation coordinates (as quaternion)
                bone.rotation_quaternion = (1, 0, 0, 0)
                # Set scale coordinates
                bone.scale = (1, 1, 1)
                # Clear any gimbal lock
                bone.rotation_mode = 'QUATERNION'
            else:
                # Set rotation coordinates (as quaternion)
                bone.rotation_quaternion = (1, 0, 0, 0)
                # Set scale coordinates
                bone.scale = (1, 1, 1) 
                # Clear any gimbal lock
                bone.rotation_mode = 'QUATERNION'
        # Update the viewport to reflect the changes
        bpy.context.view_layer.update()
        # Redraw the viewport -------------- Optional
        bpy.ops.wm.redraw_timer()      
    else:
        print("Armature named 'Petals-Movement' not found.")

def set_rose_color(collection_name, subcollection_name, color):
    # Find the main collection
    main_collection = bpy.data.collections.get(collection_name)
    
    if not main_collection:
        print(f"Collection '{collection_name}' not found.")
        return
    
    # Find the subcollection within the main collection
    sub_collection = main_collection.children.get(subcollection_name)
    
    if not sub_collection:
        print(f"Subcollection '{subcollection_name}' not found inside '{collection_name}'.")
        return
    
    sub_sub_collection_names = ["Layer-1", "Layer-2", "Layer-3", "Layer-4", "Layer-5", "Layer-6", "Layer-7", "Layer-8"]
    sub_sub_collection_list = []
    for i in sub_sub_collection_names:
        collection = sub_collection.children.get(i)
        sub_sub_collection_list.append(collection)
    
    # Set the color for all objects in the subcollection
    for i in sub_sub_collection_list:
        for obj in i.objects:
            # Ensure the object has a material slot
            if not obj.data.materials:
                # Create a new material and assign it to the object
                mat = bpy.data.materials.new(name="NewMaterial")
                obj.data.materials.append(mat)
            else:
                # Use the first material slot
                mat = obj.data.materials[0]
            
            # Ensure the material uses nodes
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            
            if bsdf:
                # Set the base color
                bsdf.inputs['Base Color'].default_value = (*color, 1.0)  # RGB + Alpha
                print(f"Changed color of '{obj.name}' to {color}.")
            else:
                print(f"Could not find Principled BSDF for '{obj.name}'.")
            # Set the viewport display color
            obj.color = (*color, 1.0)  # RGB + Alpha

def interpolate(start, end, factor):
        return start + (end - start) * factor
    
def create_all_keyframes(armature, bone, bone_1, bone_2, 
                  initial_location, initial_rotation_1, initial_rotation_2, 
                  final_location, final_rotation_1, final_rotation_2):
    # Ensure the armature is in Pose Mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    # Set the frame range for the animation
    start_frame = 1
    end_frame = 7 * 24  # Assuming 24 frames per second for 7 seconds
    # Calculate intermediate keyframe positions
    num_keyframes = 6
    frame_step = (end_frame - start_frame) // (num_keyframes - 1)
    for i in range(num_keyframes):
        frame = start_frame + i * frame_step
        factor = i / (num_keyframes - 1)  
        # Interpolated location for the bone
        location = tuple(interpolate(start, end, factor) for start, end in zip(initial_location, final_location))
        # Interpolated rotation for bone_1 and bone_2 using quaternions
        rotation_1 = tuple(interpolate(start, end, factor) for start, end in zip(initial_rotation_1, final_rotation_1))
        rotation_2 = tuple(interpolate(start, end, factor) for start, end in zip(initial_rotation_2, final_rotation_2))
        # Set bone location keyframe
        bone.location = location
        bone.keyframe_insert(data_path="location", frame=frame)
        # Set bone_1 rotation keyframe
        bone_1.rotation_mode = 'QUATERNION'
        bone_1.rotation_quaternion = rotation_1
        bone_1.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        # Set bone_2 rotation keyframe
        bone_2.rotation_mode = 'QUATERNION'
        bone_2.rotation_quaternion = rotation_2
        bone_2.keyframe_insert(data_path="rotation_quaternion", frame=frame)

def delete_all_keyframes(armature_name):
    # Get the armature object
    armature = bpy.data.objects.get(armature_name)

    if armature:
        # Check if the armature has animation data
        if armature.animation_data:
            # Clear animation data
            armature.animation_data_clear()
            print(f"Cleared animation data for {armature_name}")
        else:
            print(f"No animation data found for {armature_name}")
    else:
        print(f"Armature '{armature_name}' not found")
        
def stop_playback(scene):
    if bpy.context.scene.frame_current == bpy.context.scene.frame_end:
        bpy.ops.screen.animation_cancel(restore_frame=False)

def play_animation(armature_name):
    # Clear all frame change handlers
    bpy.app.handlers.frame_change_pre.clear()

    # Find the armature object
    armature = bpy.data.objects.get(armature_name)
    
    if not armature or not armature.animation_data or not armature.animation_data.action:
        print(f"Armature '{armature_name}' or its action not found.")
        return

    action = armature.animation_data.action

    # Set the extrapolation mode to "Constant" for all curves
    for fcurve in action.fcurves:
        fcurve.extrapolation = 'CONSTANT'
    
    # Set the current frame to the start of the animation
    bpy.context.scene.frame_set(int(action.frame_range[0]))
    
    # Set the scene's frame range to match the animation
    bpy.context.scene.frame_start = int(action.frame_range[0])
    bpy.context.scene.frame_end = int(action.frame_range[1])
    
    # Add the handler to stop the animation
    bpy.app.handlers.frame_change_pre.append(stop_playback)
    
    # Play the animation
    bpy.ops.screen.animation_play()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Loading the emotion scores prediction model, based on transformer.
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    # Loading the multi output regressor model to predict animation parameters
    with open('/Users/akhilreddy/Downloads/UF-MS/4.Spring-24/Special-Topics-Expressive-Agents/Project/Final/animation-params.pkl', 'rb') as file:
        animation_params_model = pickle.load(file)

    armature_name = "Petals-Movement"
    bone_name = "Onc-Controller"
    bone_name_1 = "Stem-Root"
    bone_name_2 = "Stem-Root-1"

    armature = bpy.data.objects.get(armature_name)
    bone_location = armature.pose.bones.get(bone_name)
    bone_rotation_1 = armature.pose.bones.get(bone_name_1)
    bone_rotation_2 = armature.pose.bones.get(bone_name_2)

    user_input_log = []
    name_input = input("May we know your name? \n")
    text_input = input(f"Hi! {name_input} How are you feeling today and what are your thoughts \n")

    while text_input != "quit":
        initial_location = bone_location.location
        initial_rotation_1 = bone_rotation_1.rotation_quaternion  
        initial_rotation_2 = bone_rotation_2.rotation_quaternion
        
        user_input_log.append(text_input)
        
        emotion_scores = emotion_classifier(text_input)
        emotion_scores = np.array([[(i["score"]*100) for i in emotion_scores[0]]])
        
        animation_parameters = animation_params_model.predict(emotion_scores)
        
        rgb = tuple(animation_parameters[0][0:3])
        final_location = tuple((animation_parameters[0][3:6]))
        final_rotation_1 = tuple(animation_parameters[0][6:10])
        final_rotation_2 = tuple(animation_parameters[0][10:14])
        
        rgb = tuple([round(i,6) for i in rgb])
        final_location = tuple([round(i,6) for i in final_location])
        final_rotation_1 = tuple([round(i,6) for i in final_rotation_1])
        final_rotation_2 = tuple([round(i,6) for i in final_rotation_2])
                
        create_all_keyframes(armature, bone_location, bone_rotation_1, bone_rotation_2, initial_location, initial_rotation_1, initial_rotation_2, final_location, final_rotation_1, final_rotation_2)
        set_rose_color("Rose", "Petals", rgb)
        play_animation(armature_name)
        # Load the machine learning model, that predicts the animation parameters based on emotion scores
        text_input = input(f"What else is on your mind {name_input}? \n")
        delete_all_keyframes(armature_name)
    
#    delete_all_keyframes(armature_name)
#    # Setting the objects of the rose to deafault position
#    set_rose_to_default_position()
#    set_rose_color("Rose", "Petals", (1.0, 0.020, 0.034))
#    
