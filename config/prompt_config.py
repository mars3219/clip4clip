prompt = dict(
    outdoor=dict(
        start=['there is'],
        pos_words=["wrestling", "kicking", "boxing", "fight", "punching",],
        neg_words=[ "jumping", "peaceful State of", "normal State of", "walking", "talking", "taking a walk", "walking side by side"],
        gender = ['man', 'woman', 'men', 'women', 'persons', 'people'],
        loc = ['at center', 'at right', 'at left', 'at top', 'at bottom', 'at top-right', 'at top-left', 'at bottom-right', 'at bottom-left', 'at side', 'in middle'],
        time_env = ['in daytime' , 'at night', "each other" ],
    ),
)


# prompt = dict(
#     outdoor=dict(
#         start=['there is', 'there are', 'a photo of', 'a scene of', ' '],
#         pos_words=["wrestling", "kicking", "boxing", "fight", "punching", "seizing", "a fistfight of", "battle of", "assult of", "group fighting of", "gang fight of"],
#         neg_words=[ "an usual state of", "normal group of", "peaceful State of", "normal State of", "a daily scene of", "a stable scene of", "a ordery scene of", "walking", "talking", "taking a walk", "walking side by side"],
#         gender = ['man', 'woman', 'men', 'women', 'persons', 'people'],
#         loc = ['at center', 'at right', 'at left', 'at top', 'at bottom', 'at top-right', 'at top-left', 'at bottom-right', 'at bottom-left', 'at side', 'in middle'],
#         time_env = ['in daytime' , 'at night' ],
#     ),
#     elevator=dict(
#         start=['there is', 'there are', 'a photo of', 'a scene of', ' '],
#         pos_words=["fight", "punching", "seizing", "a fistfight of", "battle of", "assult of", "attack of", "strike of", "hit of"],
#         neg_words=[ "an usual state of", "normal group of", "peaceful State of" ,  "normal State of", "a daily scene of" , "a stable scene of", "a ordery scene of", "walking", "talking", "standing" ],
#         gender = ['man', 'woman', 'men', 'women', 'persons', 'people'],
#         loc = ['at center', 'at right', 'at left', 'at top', 'at bottom', 'at top-right', 'at top-left', 'at bottom-right', 'at bottom-left', 'at side', 'in middle'],
#         time = ['in elevator'],
#     )
# )

# prompt_pos=[prompt['outdoor']['start'], prompt['outdoor']['pos_words'], prompt['outdoor']['gender'], prompt['outdoor']['loc'], prompt['outdoor']['time_env']]
# prompt_neg=[prompt['outdoor']['start'], prompt['outdoor']['neg_words'], prompt['outdoor']['gender'], prompt['outdoor']['loc'], prompt['outdoor']['time_env']]