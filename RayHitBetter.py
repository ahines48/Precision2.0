import bpy
import bgl
import blf
import gpu
import sys
import logging
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
import bpy_extras.view3d_utils as view3d_utils
#
# exec(open("/Users/ameliahines/PrecisionProject/RayHitBetter.py").read())
log_file = "/tmp/blender_raycast.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Logging started")
print('working')
for km in bpy.context.window_manager.keyconfigs.addon.keymaps:
    for kmi in km.keymap_items:
        print(kmi.idname, kmi.type, kmi.value)

class RaycastFromCameraViewOperator(bpy.types.Operator):
    """Click on camera view to raycast and find intersection point"""
    bl_idname = "view3d.raycast_from_camera"
    bl_label = "Raycast from Camera View"
    
    def __init__(self):
        self.draw_handle = None
        self.draw_event = None
        self.intersection_point = None
        self.click_pos = None
        self.prevIntersect = None
        self.camera_pos = None
    
    def invoke(self, context, event):
        # Only work in camera view
        logging.debug("Raycast from camera view operator invoked")
        if context.space_data.region_3d.view_perspective != 'CAMERA':
            logging.debug("Please enter camera view first")
            self.report({'WARNING'}, "Please enter camera view first")
            return {'CANCELLED'}
        
        # Get the click position in normalized coordinates (0-1)
        self.click_pos = (event.mouse_region_x, event.mouse_region_y)
        logging.debug(f"Mouse Click at: {self.click_pos}")
        # Perform the raycast
        self.raycast(context)
        
        # Set up drawing
        if self.intersection_point is not None and self.prevIntersect is None:
            logging.debug("Intersection found, Calling Draw.")
            self.create_visualization(context)
            context.window_manager.modal_handler_add(self)
            self.prevIntersect = self.intersection_point
            return {'RUNNING_MODAL'}
        elif (self.intersection_point-self.prevIntersect).length > .0000001:
            logging.debug("New Intersection found, proceeding to setup draw.")
            self.create_visualization(context)
            context.window_manager.modal_handler_add(self)
            self.prevIntersect = self.intersection_point
            return {'RUNNING_MODAL'}
        elif (self.intersection_point-self.prevIntersect).length < .0000001:
            logging.debug("Intersection found, but too close to previous intersection.")
            logging.debug(f"Distance: {(self.intersection_point-self.prevIntersect).length}")
            return {'CANCELLED'}
        else:
            self.report({'INFO'}, "No intersection found")
            return {'CANCELLED'}
        
    def create_visualization(self, context):
        """Create mesh objects for visualization"""
        # Clear previous visualization
        if self.intersection_point and self.camera_pos:
            self.line_obj = create_line_object(
                "RaycastLine",
                self.camera_pos,
                self.intersection_point
            )
            self.point_obj = create_point_object(
                "IntersectionPoint",
                self.intersection_point
            )
    
    def clear_visualization(self):
        """Remove visualization objects"""
        if self.line_obj:
            bpy.data.objects.remove(self.line_obj, do_unlink=True)
        if self.point_obj:
            bpy.data.objects.remove(self.point_obj, do_unlink=True)
        self.line_obj = None
        self.point_obj = None

    def raycast(self, context):
        # Get camera and scene
        logging.debug("Raycasting function called")
        camera = context.scene.camera
        self.camera_pos = camera.matrix_world.translation
        scene = context.scene
        
        # Get the region and space data
        region = context.region
        rv3d = context.region_data
        
        # Convert mouse position to 3D view coordinates
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, self.click_pos)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self.click_pos)
        ray_direction = view_vector.normalized()
        
        # Perform the raycast
        success, location, normal, index, obj, matrix = scene.ray_cast(
            depsgraph=context.evaluated_depsgraph_get(),
            origin=ray_origin,
            direction=ray_direction,
            distance=400,
        )
        logging.debug(f"Ray Origin: {ray_origin}")
        logging.debug(f"Ray Direction: {ray_direction}")
        if success:
            self.intersection_point = location
            logging.debug(f"Intersection at: {location}")
            output_path = "/Users/ameliahines/Desktop/Precision1.0_1613/1613_OptLocs/OptLocs_260520_1613_S10.txt"
            with open(output_path, "a") as f:
                f.write(f"{location.x}, {location.y}, {location.z}\n")
        else:
            self.intersection_point = None
            logging.debug("Raycast failed.")
            
    
    def modal(self, context, event):
        # logging.debug(f"Modal function called, event: {event.type}")
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'CANCELLED'} 
        
        return {'PASS_THROUGH'}
    
    def cancel(self, context):
        if self.draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self.draw_handle, 'WINDOW')
            self.draw_handle = None
        
def register():
    logging.debug("Raycast from camera view operator registered")
    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name='3D View', space_type='VIEW_3D')
    kmi = km.keymap_items.new(RaycastFromCameraViewOperator.bl_idname, 'LEFTMOUSE', 'DOUBLE_CLICK', ctrl=True)
    kmi.active = True
    bpy.utils.register_class(RaycastFromCameraViewOperator)
    
def unregister():
    wm = bpy.context.window_manager
    for kc in wm.keyconfigs:
        km = kc.keymaps.get('3D View')
        if km:
            for kmi in km.keymap_items:
                if kmi.idname == RaycastFromCameraViewOperator.bl_idname:
                    km.keymap_items.remove(kmi)
                    break
    
    # 3. Force remove any remaining instances
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            for region in area.regions:
                if region.type == 'WINDOW':
                    region.tag_redraw()
    bpy.utils.unregister_class(RaycastFromCameraViewOperator)

def create_point_object(name, location, color=(0, 1, 0, 1), size=0.002):
    """Create a point as a small sphere"""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    
    # Set material color
    mat = bpy.data.materials.new(name="PointMaterial")
    mat.diffuse_color = color
    obj.data.materials.append(mat)
    
    return obj

def create_line_object(name, start, end, color=(1, 0, 0, 1)):
    """Create a line as a mesh object"""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Create the line
    mesh.from_pydata([start, end], [(0, 1)], [])
    mesh.update()
    
    # Set material color
    mat = bpy.data.materials.new(name="LineMaterial")
    mat.diffuse_color = color
    obj.data.materials.append(mat)
    
    return obj

if __name__ == "__main__":
    register()

    
# To use:
# 1. Enter camera view in Layout tab
# 2. Press Ctrl+LeftClick on a point in the camera view
# 3. The script will draw a red line from the camera to the intersection point
# 4. The coordinates will be printed to the console
# 5. Right-click or Esc to clear the visualization
