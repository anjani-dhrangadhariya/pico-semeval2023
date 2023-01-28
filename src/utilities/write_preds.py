import pandas as pd


picos_mapping = {'participant': 'POP', 'intervention': 'INT', 'outcome':'OUT'}

def preds2labs(labs, args):

    actual_labs = []

    for l in labs:
        if l == 0 or l == 0.0:
            actual_labs.append( 'O' )
        else:
            actual_labs.append( picos_mapping[args.entity] )

    return actual_labs

def flatten_df( df, args ):

    post_ids = []
    subreddit_ids = []
    word_list = []
    label_list = []


    for i, x in df.to_dict('index').items():
        
        post_id = x['post_id']
        post_id = [post_id] * len( x['predictions'] )

        subreddit_id = x['subreddit_id']
        subreddit_id = [subreddit_id] * len( x['predictions'] )

        words = x['tokens']
        labels = x['predictions']

        post_ids.extend(post_id) 
        subreddit_ids.extend(subreddit_id) 
        word_list.extend(words) 

        labels = preds2labs(labels, args)
        label_list.extend(labels)

    L = {'post_ids':post_ids, 'subreddit_id':subreddit_ids, 'words': word_list, 'labels':label_list }
    flattened_df = pd.DataFrame(L)

    return flattened_df


def write_preds(df, exp_args, last_key):

    epoch_i = last_key.split('_')[-1]

    base_path = f"/mnt/nas2/results/Results/systematicReview/SemEval2023/predictions_test/without_Dropout"
    file_name_here = base_path + '/' + str(exp_args.entity) + '/' + str(exp_args.seed) + '/' + str(exp_args.embed) + '/' + str(exp_args.model) + '_' + str(exp_args.predictor) + '_ep_' + str(epoch_i) + '.csv'
    df.to_csv(file_name_here, encoding='utf-8')