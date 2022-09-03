import numpy as np
from tLinkage.b_tlinkage import evaluation

if __name__ == '__main__':
    # region INPUT
    mode = 2  # [1 for Motion Segmentation | 2 for Plane Segmentation]
    algorythms = ["base","tlinkage","tlinkage-gmart","tlinkage-pac","multilink","multilink-gmart"]
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
        t_link_err = []
        t_link_gmart_dyn_err = []
        t_link_gmart_cos_err = []
        images_err_list = []
        for l in range(len(labels)):
            images_err_list.append([])

        a = []
        b = []
        c = []
        for par in range(len(parameters)):
            print("Parameters : ", parameters[par])
            for k in range(len(labels)):
                print('Image number: ', k)
                t_link_err.clear()
                t_link_gmart_dyn_err.clear()
                t_link_gmart_cos_err.clear()

                for i in range(iterations):
                    #print('Iteration number: ', i)
                    errors_list = evaluation(tau[k], labels[k], "FM",
                                            OUTLIER_THRESHOLD_GMART=parameters[par][0],
                                            NUMBER_OF_CLUSTER=parameters[par][1],
                                            algorythms=algorythms)
                    t_link_err.append(errors_list[0])
                    t_link_gmart_dyn_err.append(errors_list[1])
                    t_link_gmart_cos_err.append(errors_list[2])

                '''print('Mean error for t_linkage is: ', np.sum(t_link_err) / iterations)
                print('Mean error for t_linkage + gmart_dyn is: ', np.sum(t_link_gmart_dyn_err) / iterations)
                print('Mean error for t_linkage gmart_cos is: ', np.sum(t_link_gmart_cos_err) / iterations)
                print('\n')'''

                images_err_list[k].append(np.sum(t_link_err) / iterations)
                images_err_list[k].append(np.sum(t_link_gmart_dyn_err) / iterations)
                images_err_list[k].append(np.sum(t_link_gmart_cos_err) / iterations)

            a.clear()
            b.clear()
            c.clear()

            for i in range(np.shape(images_err_list)[0]):
                a.append(images_err_list[i][0])
                b.append(images_err_list[i][1])
                c.append(images_err_list[i][2])

            print('Mean error for t_linkage for the entire dataset is: ', np.sum(a) / len(images_err_list))
            print('Mean error for t_linkage + gmart_dyn for the entire dataset is: ', np.sum(b) / len(images_err_list))
            print('Mean error for t_linkage gmart_cos for the entire dataset is: ', np.sum(c) / len(images_err_list))

            for l in range(len(labels)):
                images_err_list[l].clear()

    # endregion
    if mode == 2:

        # region [CASE 2] Plane segmentation
        labels = ['barrsmith', 'bonython', 'elderhalla', 'elderhallb', 'hartley',
                  'ladysymon', 'library', 'napiera', 'napierb', 'neem', 'nese', 'oldclassicswing', 'physics', 'sene',
                  'unionhouse'
                  ]

        tau = [60, 50, 50, 50, 15, 25, 20, 35, 25, 25, 25, 20, 4, 25, 10, 50, 35, 50, 35]
        parameters = [[35,90],[32,90],[27,90],[25,90]]

        t_link_err = []
        t_link_gmart_dyn_err = []
        t_link_gmart_cos_err = []
        images_err_list = []
        for l in range(len(labels)):
            images_err_list.append([])

        a = []
        b = []
        c = []
        for par in range(len(parameters)):
            print("Parameters : ", parameters[par])

            for k in range(len(labels)):
                print('Image number: ', k)

                for i in range(iterations):
                    #print('Iteration number: ', i)
                    errors_list = evaluation(tau[k], labels[k], "H",
                                            OUTLIER_THRESHOLD_GMART=parameters[par][0],
                                            NUMBER_OF_CLUSTER=parameters[par][1],
                                            algorythms=algorythms)
                    t_link_err.append(errors_list[0])
                    t_link_gmart_dyn_err.append(errors_list[1])
                    t_link_gmart_cos_err.append(errors_list[2])

                '''print('Mean error for t_linkage is: ', np.sum(t_link_err) / iterations)
                print('Mean error for t_linkage + gmart_dyn is: ', np.sum(t_link_gmart_dyn_err) / iterations)
                print('Mean error for t_linkage gmart_cos is: ', np.sum(t_link_gmart_cos_err) / iterations)
                print('\n')'''

                images_err_list[k].append(np.sum(t_link_err) / iterations)
                images_err_list[k].append(np.sum(t_link_gmart_dyn_err) / iterations)
                images_err_list[k].append(np.sum(t_link_gmart_cos_err) / iterations)

                t_link_err.clear()
                t_link_gmart_dyn_err.clear()
                t_link_gmart_cos_err.clear()

            a.clear()
            b.clear()
            c.clear()

            for i in range(np.shape(images_err_list)[0]):
                a.append(images_err_list[i][0])
                b.append(images_err_list[i][1])
                c.append(images_err_list[i][2])

            print('Mean error for t_linkage for the entire dataset is: ', np.sum(a) / len(images_err_list))
            print('Mean error for t_linkage + gmart_dyn for the entire dataset is: ', np.sum(b) / len(images_err_list))
            print('Mean error for t_linkage gmart_cos for the entire dataset is: ', np.sum(c) / len(images_err_list))

            for l in range(len(labels)):
                images_err_list[l].clear()