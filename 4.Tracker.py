import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # Each time a new object ID is detected, the count increases by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and IDs
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected; assign a new ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by removing unused IDs
        new_center_points = {obj_bb_id[4]: self.center_points[obj_bb_id[4]] for obj_bb_id in objects_bbs_ids}

        # Update dictionary with only active IDs
        self.center_points = new_center_points.copy()

        return objects_bbs_ids
