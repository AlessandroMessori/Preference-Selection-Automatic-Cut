import numpy as np
from tLinkage.b_tlinkage import evaluation

if __name__ == '__main__':
    # region INPUT
    mode = 2  # [1 for Motion Segmentation | 2 for Plane Segmentation]
    algorythms = ["base","tlinkage","tlinkage-gmart","multilink","multilink-gmart"]
    errors = ["tlinkage","tlinkage-gmart-dyn","tlinkage-gmart-cos","multilink","multilink-gmart-dyn","multilink-gmart-cos"]
    iterations = 5
    # endregion
    if mode == 1:
        # region [CASE 1] Motion segmentation
        labels = ['biscuit', 'biscuitbook', 'biscuitbookbox', 'boardgame', 'book', 'breadcartoychips',
                  'breadcube', 'breadcubechips', 'breadtoy', 'breadtoycar', 'carchipscube', 'cube', 'cubebreadtoychips',
                  'cubechips', 'cubetoy', 'dinobooks', 'game', 'gamebiscuit', 'toycubecar'
                  ]

        tau = [14, 14, 3, 30, 14, 30, 14, 14, 14, 30, 14, 14, 14, 14, 14, 30, 14, 14, 14]
        parameters = [[40, 50], [40, 40], [40, 30], [40, 20], [40, 15], [40, 10], [40, 5], [40, 2]]
        err_dict = {}
        images_err_list = []
        for l in range(len(labels)):
            images_err_list.append({})

        for par in range(len(parameters)):
            print("Parameters : ", parameters[par])
            for k in range(len(labels)):
                print('Image number: ', k)
                err_dict = dict()

                for i in range(iterations):
                    #print('Iteration number: ', i)
                    curr_errors_dict = evaluation(tau[k], labels[k], "FM",
                                            OUTLIER_THRESHOLD_GMART=parameters[par][0],
                                            NUMBER_OF_CLUSTER=parameters[par][1],
                                            algorythms=algorythms)

                    for key in curr_errors_dict:
                        if key in err_dict:
                            err_dict[key].append(curr_errors_dict[key])
                        else:
                            err_dict[key] = [curr_errors_dict[key]]

                for key in err_dict:
                    err = np.sum(err_dict[key]) / iterations
                    print('Mean error for ', key, ' is: ', np.round(err,4))
                    images_err_list[k][key] = err
                print('\n')

            fullErrors = dict()
            for key in errors:
                fullErrors[key] = []

            for i in range(np.shape(images_err_list)[0]):
                for key in errors:
                    fullErrors[key].append(images_err_list[i][key])

            for key in err_dict:
                    print('Mean error for ', key,' for the entire dataset is: ', np.round(np.sum(fullErrors[key]) / len(images_err_list),4))
            print('\n')

            for l in range(len(labels)):
                images_err_list[l] = dict()

    # endregion
    if mode == 2:

        # region [CASE 2] Plane segmentation
        labels = ['barrsmith', 'bonython', 'elderhalla', 'elderhallb', 'hartley',
                  'ladysymon', 'library', 'napiera', 'napierb', 'neem', 'nese', 'oldclassicswing', 'physics', 'sene',
                  'unionhouse'
                  ]

        tau = [60, 50, 50, 50, 15, 25, 20, 35, 25, 25, 25, 20, 4, 25, 10, 50, 35, 50, 35]
        parameters = [[35,90],[32,90],[27,90],[25,90]]

        err_dict = {}
        images_err_list = []
        for l in range(len(labels)):
            images_err_list.append({})
            
        for par in range(len(parameters)):
            print("Parameters : ", parameters[par])
            for k in range(len(labels)):
                print('Image number: ', k)
                err_dict = dict()

                for i in range(iterations):
                    #print('Iteration number: ', i)
                    curr_errors_dict = evaluation(tau[k], labels[k], "H",
                                            OUTLIER_THRESHOLD_GMART=parameters[par][0],
                                            NUMBER_OF_CLUSTER=parameters[par][1],
                                            algorythms=algorythms)

                    for key in curr_errors_dict:
                        if key in err_dict:
                            err_dict[key].append(curr_errors_dict[key])
                        else:
                            err_dict[key] = [curr_errors_dict[key]]

                for key in err_dict:
                    err = np.sum(err_dict[key]) / iterations
                    print('Mean error for ', key, ' is: ', np.round(err,4))
                    images_err_list[k][key] = err
                print('\n')

            fullErrors = dict()
            for key in errors:
                fullErrors[key] = []

            for i in range(np.shape(images_err_list)[0]):
                for key in errors:
                    fullErrors[key].append(images_err_list[i][key])

            for key in err_dict:
                    print('Mean error for ', key,' for the entire dataset is: ', np.round(np.sum(fullErrors[key]) / len(images_err_list),4))
            print('\n')

            for l in range(len(labels)):
                images_err_list[l] = dict()
