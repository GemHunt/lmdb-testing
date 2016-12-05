from collections import namedtuple
import cPickle as pickle
import cv2
import caffe_image as ci
import pandas as pd
import graph

results_dict = {}
Image = namedtuple('Image','seed_image_id image_id angle max_value')
Group = namedtuple('Group','group_id starting_seed_id images')
seed_groups = []

def read_results(cut_off,data_dir,seed_image_ids = []):
    all_results = pickle.load(open(data_dir + 'all_results.pickle', "rb"))
    # columns = ['seed_image_id', 'image_id', 'angle', 'max_value']
    image_ids_with_highest_max_value = {}

    # This fills image_ids_with_highest_max_value:
    for results in all_results:
        for seed_image_id, image_id, angle, max_value in results:
            # Well, we know this was a match already:
            if seed_image_id == image_id:
                continue
            # This optionally filters the results smaller:
            if len(seed_image_ids) != 0 and seed_image_id not in seed_image_ids:
                continue
            # This optionally filters only the best results:
            if max_value < cut_off:
                continue

            if image_id in image_ids_with_highest_max_value:
                if image_ids_with_highest_max_value[image_id][2] < max_value:
                    image_ids_with_highest_max_value[image_id] = [seed_image_id, angle, max_value]
            else:
                image_ids_with_highest_max_value[image_id] = [seed_image_id, angle, max_value]

            if not seed_image_id in results_dict:
                results_dict[seed_image_id] = {}

            if not image_id in results_dict[seed_image_id]:
                results_dict[seed_image_id][image_id] = [max_value, angle]

            if max_value > results_dict[seed_image_id][image_id][0]:
                results_dict[seed_image_id][image_id] = [max_value, angle]

    pickle.dump(results_dict, open(data_dir + 'seed_data.pickle', "wb"))

def get_results_list(seed_id_filter = -1):
    results_list = []
    for seed_image_id, seed_values in results_dict.iteritems():
        if seed_id_filter == -1:
            pass
        else:
            if seed_image_id != seed_id_filter:
                continue
        for image_id, values in seed_values.iteritems():
            max_value, angle = values
            Image = namedtuple('Image', 'seed_image_id image_id angle max_value')
            results_list.append(Image(seed_image_id, image_id, angle, max_value))
    return results_list


def set_test_images_to_be_exclusive_in_seeds():
    #no dups:
    #if many_image_ids_per_seed_ok == False:
        #seeds = []
        #for key, values in image_ids_with_highest_max_value.iteritems():
            #if not values[0] in seeds:
                #seeds[values[0]] = []
            #seeds[values[0]].append([values[2], values[1], key])
    pass

def set_angles_postive():
    Image = namedtuple('Image', 'seed_image_id image_id angle max_value')
    results_list = get_results_list()
    new_results_list = []

    for image in results_list:
        #Flip the seeds and test image_ids on all 331-359 angles so all angles are 0-29.
        if image.angle < 30:
            new_results_list.append(image)
        else:
            seed_image_id = image.image_id
            image_id = image.seed_image_id
            angle = 360 - image.angle
            max_value = image.max_value
            new_results_list.append(Image(seed_image_id,image_id,angle,max_value))

    results_list = new_results_list
    for image in results_list:
        if not image.seed_image_id in results_dict:
            results_dict[image.seed_image_id] = {}

        if not image.image_id in results_dict[image.seed_image_id]:
            results_dict[image.seed_image_id][image.image_id] = [image.max_value, image.angle]
        else:
            existing_max_value = results_dict[image.seed_image_id][image.image_id][0]
            existing_angle = results_dict[image.seed_image_id][image.image_id][1]
            if abs(image.angle - existing_angle) > 3:
                print 'Angles off by more than 3: ',image, existing_angle

            if image.max_value > existing_max_value:
                results_dict[image.seed_image_id][image.image_id] = [image.max_value, image.angle]

def create_group():
    #Set the first top-link for the starting seed will be the test image with the most points.
    #Keep going until I get back to the starting seed.
    pass

def set_starting_seed():
    results_left = results_dict.copy()
    df = pd.DataFrame(get_results_list())
    grouped = df.groupby(by=['seed_image_id'])['max_value'].sum()
    starting_seed_id = grouped.idxmax()

    links = pd.DataFrame(get_results_list(starting_seed_id))
    grouped = df.groupby(by=['seed_image_id'])['max_value'].sum()
    test_id = grouped.idxmax()
    del results_left[starting_seed_id][test_id]


    #The seed left with the most points(sum of max_value for all links) is the starting seed.
    #Starting seed is 0 and 360.
    pass

def get_nodes():
    nodes = set(results_dict.keys())
    for seed_image_id, seed_values in results_dict.iteritems():
        for image_id, values in seed_values.iteritems():
            if not image_id in nodes:
                nodes.add(image_id)
    return nodes

def get_edges():
    # this is currently ignoring dup edges.
    edges = {}
    for seed_image_id, seed_values in results_dict.iteritems():
        for image_id, values in seed_values.iteritems():
            max_value = values[0]
            angle = values[1]
            node1 = seed_image_id
            node2 = image_id
            edge_value = [max_value,angle]

            #flip the node order so the first node image_id is always lower
            if node1 < node2:
                temp = node1
                node1 = node2
                node2 = temp
                edge_value = [max_value,-angle]
            edge_key = (node1,node2)
            if not edge_key in edges.keys():
                edges[edge_key] = edge_value
    return edges

def drop_bad_nodes(data_dir, seed_image_id):
    nodes = pickle.load(open(data_dir + 'nodes.pickle', "rb"))
    edges = pickle.load(open(data_dir + 'edges.pickle', "rb"))

    test_images = results_dict[seed_image_id]

    paths = graph.get_paths(nodes, edges.keys(), seed_image_id, test_images.keys())
    #Get total_short_paths, total_good_short_paths
    #Get the correct angle first.


    graph_results = []

    for edge_paths in paths:
        max_value_ave = 0
        test_image_id = 0
        test_max_value = 0
        test_image_angle = 0
        angles = []
        max_value_path_total = 0

        for path in edge_paths:
            node1 = -1
            node2 = -1
            angle_total = 0
            max_value_edge_path_total = 0

            for node in path:
                if node1 == -1:
                    node1 = node
                    continue
                node2 = node
                key = (node1, node2)
                if key in edges:
                    max_value, angle = edges[(node1, node2)]
                else:
                    max_value, angle = edges[(node2, node1)]
                    angle = -angle
                #print node1, node2, max_value, angle
                angle_total += angle
                max_value_edge_path_total += max_value
                node1 = node

            max_value_path_total += max_value_edge_path_total / len(path)
            angle_total = ci.get_pos_angle(angle_total)
            if len(path) == 2:
                test_image_id = node2
                test_max_value = max_value
                test_image_angle = angle_total
                #print '                       ', path, angle_total, '\n'
            else:
                angles.append(angle_total)
                #print '    ', path, angle_total, '\n'
        good_paths_count = 0
        for angle in angles:
            if abs(test_image_angle - angle) < 4:
                good_paths_count +=1
        max_value_ave = max_value_path_total/len(edge_paths)
        graph_results.append([test_image_id, test_image_angle, test_max_value, max_value_ave, len(edge_paths), good_paths_count])

    graph_results = sorted(graph_results, key=lambda graph_results: graph_results[2], reverse=True)

    widened_seeds = [3893,5107,6280,9813]
    max_positive_connections = 0
    most_positive_connected_test_image_id = -1
    for result in graph_results:
        print result
        #if 330 < result[1]:
        if 10 < result[1] < 30:
            if result[5] > max_positive_connections:
                if result[0] not in widened_seeds:
                    most_positive_connected_test_image_id = result[0]
                    max_positive_connections = result[5]
    print '\nmost_positive_connected_test_image_id:',most_positive_connected_test_image_id,'max_positive_connections:',max_positive_connections

    graph_results = [x for x in graph_results if x[0] == most_positive_connected_test_image_id]
    print 'test_image_id, test_image_angle, test_max_value, max_value_ave, edge_path_count, good_paths_count'
    print graph_results
    return most_positive_connected_test_image_id, graph_results[0][1]


def create_composite_images(crop_dir,data_dir,crop_size,rows,cols, seed_image_id_filter = -1):
    graph = []
    results = {}
    if seed_image_id_filter == -1:
        results = results_dict
    else:
        results[seed_image_id_filter] = results_dict[seed_image_id_filter]

    for seed_image_id, seed_values in results.iteritems():
        images = []
        crop_size = 160
        images.append(ci.get_rotated_crop(crop_dir,seed_image_id, crop_size, 0))

        results = []
        for image_id, values in seed_values.iteritems():
            max_value, angle = values
            results.append([image_id, max_value,angle])

        sorted_results = sorted(results, key=lambda result: result[1], reverse=True)
        for image_id, max_value,angle in sorted_results:
            #print str(seed_image_id) + '\t' + str(image_id)  + '\t' + str(max_value)  + '\t' + str(angle)
            #values.sort(key=lambda x: x[0], reverse=True)
            crop = ci.get_rotated_crop(crop_dir,image_id, crop_size, angle)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(crop, str(max_value)[0:5], (10, 20), font, .7, (0, 255, 0), 2)
            cv2.putText(crop, str(image_id)[0:5], (10, 90), font, .7, (0, 255, 0), 2)
            images.append(crop)
        composite_image = ci.get_composite_image(images,rows,cols)
        cv2.imwrite(data_dir + str(seed_image_id) + '.png', composite_image)


