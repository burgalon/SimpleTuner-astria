import json

JOB_STR_1 = '{"id":2002368,"name":"man","created_at":"2025-01-05T08:58:31.696Z","updated_at":"2025-05-25T14:26:26.521Z","user_id":2,"trained_at":"2025-01-05T09:09:14.690Z","started_training_at":"2025-05-25T14:26:26.520Z","steps":1000,"title":"Alon portrait","branch":"flux1","callback":None,"process_ip":"akash-wqtj5-2,3","trials":31,"num_prompts":0,"is_api":False,"base_tune_id":1504944,"token":"ohwx","args":"preset=flux-lora-portrait learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial flux_lora_target=portrait segmentation=1 only_face=true","cost":None,"expires_at":"2025-06-04T09:09:14.690Z","emailed_notice":False,"public_at":None,"face_crop":True,"checkpoint_deleted":False,"checkpoint_deleted_at":None,"failed_at":None,"model_type":"lora","sha256":None,"model_url":None,"description_url":None,"cost_mc":216000,"training_face_correct":False,"eta":"2025-01-05T09:08:36.253Z","base_pack_id":260,"characteristics":{"age":"30 yo","ethnicity":"hispanic","eye_color":"brown eyes","facial_hair":"","glasses":"","hair_color":"black hair","hair_length":"short hair","hair_style":"bald","headcover":"","is_bald":"bald","name":"man"},"prompts_callback":None,"auto_extend":False,"orig_images":["https://sdbooth2-production.s3.amazonaws.com/ygesi07jlrw2nk31pq3tfu5vztgf","https://sdbooth2-production.s3.amazonaws.com/viatl4d53mu1ohsyymfu3hb6y74z","https://sdbooth2-production.s3.amazonaws.com/2mkj9khd3n2gm6usn0ppcnxxnyw6","https://sdbooth2-production.s3.amazonaws.com/2o99s9pxtny5ohrx0s1zp2wakha3","https://sdbooth2-production.s3.amazonaws.com/kjn9jgzchm4sl3uj1mo2zh2zv72f","https://sdbooth2-production.s3.amazonaws.com/3xfxajtel9jwq1al1fe8mey5rh17","https://sdbooth2-production.s3.amazonaws.com/4bp5gnc52qzbo2pj2nzwondpekui"],"file_names":[{"filename":"2xuxve6e6bh50w6sh97avcurxkeb.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/ygesi07jlrw2nk31pq3tfu5vztgf"},{"filename":"AA991103-E7DE-47C1-ABAB-82E794C150BF.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/viatl4d53mu1ohsyymfu3hb6y74z"},{"filename":"66gr44co9wa17nis5l5dnt3mg7gl.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/2mkj9khd3n2gm6usn0ppcnxxnyw6"},{"filename":"AA991103-E7DE-47C1-ABAB-82E794C150BF.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/2o99s9pxtny5ohrx0s1zp2wakha3"},{"filename":"IMG_8003.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/kjn9jgzchm4sl3uj1mo2zh2zv72f"},{"filename":"IMG_7988.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/3xfxajtel9jwq1al1fe8mey5rh17"},{"filename":"IMG_7659.jpeg","url":"https://sdbooth2-production.s3.amazonaws.com/4bp5gnc52qzbo2pj2nzwondpekui"}],"resolution":None,"user":{"create_ckpt":False,"backend_version":null},"prompts":[]}'
JOB_STR_2 = json.dumps({
        "id": 1858416,
        "name": "woman",
        "created_at": "2024-12-03T08:40:18.994Z",
        "updated_at": "2024-12-11T12:27:23.064Z",
        "user_id": 2,
        "trained_at": "2024-12-06T09:36:00.000Z",
        "started_training_at": "2024-12-11T12:27:23.063Z",
        "steps": 1000,
        "title": "irit preset=portrait",
        "branch": "flux1",
        "callback": None,
        "process_ip": "akash-19180702-0",
        "trials": 19,
        "num_prompts": 0,
        "is_api": False,
        "base_tune_id": 1504944,
        "token": "ohwx",
        "args": "preset=flux-lora-portrait learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial flux_lora_target=portrait segmentation=1 only_face=true",
        "cost": None,
        "expires_at": "2025-01-05T09:36:00.000Z",
        "emailed_notice": False,
        "public_at": None,
        "face_crop": True,
        "checkpoint_deleted": False,
        "checkpoint_deleted_at": None,
        "failed_at": "2024-12-03T10:20:01.543Z",
        "model_type": "lora",
        "sha256": "",
        "model_url": "",
        "description_url": "",
        "cost_mc": 0,
        "training_face_correct": False,
        "eta": "2024-12-06T10:00:55.196Z",
        "base_pack_id": None,
        "characteristics": None,
        "prompts_callback": None,
        "orig_images": [
        "https://sdbooth2-production.s3.amazonaws.com/jwaiky6g9b0ryqadm9gryz6wkykg",
        "https://sdbooth2-production.s3.amazonaws.com/uhtm480bwovnt4yi92eswipcjwqj",
        "https://sdbooth2-production.s3.amazonaws.com/hn2s2c5y461a5ve6y1266xgcc04c",
        "https://sdbooth2-production.s3.amazonaws.com/4skf7x147wrf9hjp61t8nvmrn9kj",
        "https://sdbooth2-production.s3.amazonaws.com/7jfx87xfrsl9hpp1jiiw3i2l440s",
        "https://sdbooth2-production.s3.amazonaws.com/37ebu983vz9kyeecjsg82f3wj23c",
        "https://sdbooth2-production.s3.amazonaws.com/cnt2pm2k4lg4933uy1w90fdr2jth",
        "https://sdbooth2-production.s3.amazonaws.com/6weh50yrdjly9utdwekemdmupocq",
        "https://sdbooth2-production.s3.amazonaws.com/bj8w0bwbrurt5cqzhsp99x27ffjc",
        "https://sdbooth2-production.s3.amazonaws.com/nvtmizkfjqh1hyfrgsjwoqq09zle",
        "https://sdbooth2-production.s3.amazonaws.com/4xvtx0n6529m6t8qhbzaz9xmekul",
        "https://sdbooth2-production.s3.amazonaws.com/602vsmswolbe9k2xde1if8uc0is4",
        "https://sdbooth2-production.s3.amazonaws.com/jn5ca51yn64u8mpx0eieqyeh3kid",
        "https://sdbooth2-production.s3.amazonaws.com/fiwba1158oocltizejyff86duhn9",
        "https://sdbooth2-production.s3.amazonaws.com/oomcupi58rxezfthrcx5d3oljj8x"
        ],
        "file_names": [
        {
            "filename": "ohwx (5).png",
            "url": "https://sdbooth2-production.s3.amazonaws.com/jwaiky6g9b0ryqadm9gryz6wkykg"
        },
        {
            "filename": "ohwx (15).JPG",
            "url": "https://sdbooth2-production.s3.amazonaws.com/uhtm480bwovnt4yi92eswipcjwqj"
        },
        {
            "filename": "ohwx (8).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/hn2s2c5y461a5ve6y1266xgcc04c"
        },
        {
            "filename": "ohwx (6).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/4skf7x147wrf9hjp61t8nvmrn9kj"
        },
        {
            "filename": "ohwx (7).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/7jfx87xfrsl9hpp1jiiw3i2l440s"
        },
        {
            "filename": "ohwx (1).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/37ebu983vz9kyeecjsg82f3wj23c"
        },
        {
            "filename": "ohwx (9).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/cnt2pm2k4lg4933uy1w90fdr2jth"
        },
        {
            "filename": "ohwx (14).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/6weh50yrdjly9utdwekemdmupocq"
        },
        {
            "filename": "ohwx (12).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/bj8w0bwbrurt5cqzhsp99x27ffjc"
        },
        {
            "filename": "ohwx (11).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/nvtmizkfjqh1hyfrgsjwoqq09zle"
        },
        {
            "filename": "ohwx (3).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/4xvtx0n6529m6t8qhbzaz9xmekul"
        },
        {
            "filename": "ohwx (10).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/602vsmswolbe9k2xde1if8uc0is4"
        },
        {
            "filename": "ohwx (4).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/jn5ca51yn64u8mpx0eieqyeh3kid"
        },
        {
            "filename": "ohwx (2).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/fiwba1158oocltizejyff86duhn9"
        },
        {
            "filename": "ohwx (13).jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/oomcupi58rxezfthrcx5d3oljj8x"
        }
        ],
        "resolution": None,
        "user": {
        "create_ckpt": False,
        "backend_version": None
        },
        "prompts": []
    })

JOB_STR_3 = json.dumps({
    "id": 2696107,
    "name": "man",
    "created_at": "2025-06-04T09:43:33.158Z",
    "updated_at": "2025-07-06T05:58:25.669Z",
    "user_id": 2,
    "trained_at": "2025-06-04T09:53:50.044Z",
    "started_training_at": "2025-07-06T05:58:25.669Z",
    "steps": 300,
    "title": "alon Jun/2026",
    "branch": "flux1",
    "callback": None,
    "process_ip": "akash-22266362-0",
    "trials": 183,
    "num_prompts": 0,
    "is_api": False,
    "base_tune_id": 1504944,
    "token": "ohwx",
    "args": "preset=flux-lora-portrait learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial flux_lora_target=portrait segmentation=1 only_face=true",
    "cost": None,
    "expires_at": "2025-08-03T09:53:50.044Z",
    "emailed_notice": False,
    "public_at": None,
    "face_crop": True,
    "checkpoint_deleted": False,
    "checkpoint_deleted_at": None,
    "failed_at": None,
    "model_type": "lora",
    "sha256": None,
    "model_url": None,
    "description_url": None,
    "cost_mc": 150000,
    "training_face_correct": False,
    "eta": "2025-06-04T09:53:33.943Z",
    "base_pack_id": None,
    "characteristics": None,
    "prompts_callback": None,
    "auto_extend": False,
    "orig_images": [
      "https://sdbooth2-production.s3.amazonaws.com/wd67ld1jyafk3t580cc4s47itzo6",
      "https://sdbooth2-production.s3.amazonaws.com/t95eam4wepwm4xl2ejt9q3w9plvs",
      "https://sdbooth2-production.s3.amazonaws.com/66jfb0q1z7zhyb16yw0ornjqny7y",
      "https://sdbooth2-production.s3.amazonaws.com/oz7g51yxnau490lakgau2o1b70nq"
    ],
    "file_names": [
      {
        "filename": "lueurph70qqzuqs7zl399mbrqsx1.jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/wd67ld1jyafk3t580cc4s47itzo6"
      },
      {
        "filename": "tqqaynhcu3ne5onxg1jsrqpvtzv4.jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/t95eam4wepwm4xl2ejt9q3w9plvs"
      },
      {
        "filename": "66gr44co9wa17nis5l5dnt3mg7gl.jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/66jfb0q1z7zhyb16yw0ornjqny7y"
      },
      {
        "filename": "mlgrx6bfadoo20c17oj8u50h70ol.jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/oz7g51yxnau490lakgau2o1b70nq"
      }
    ],
    "resolution": None,
    "user": {
      "create_ckpt": False,
      "backend_version": None
    },
    "prompts": []
  })

JOB_STR_4 = json.dumps({
    "id": 1689634,
    "name": "woman",
    "created_at": "2024-10-06T10:01:57.918Z",
    "updated_at": "2024-12-06T08:00:39.451Z",
    "user_id": 2,
    "trained_at": "2024-10-06T10:43:45.717Z",
    "started_training_at": "2024-11-06T13:13:47.860Z",
    "steps": 405,
    "title": "irit fast 5e-5",
    "branch": "flux1",
    "callback": None,
    "process_ip": "akash-18719476",
    "trials": 3,
    "num_prompts": 0,
    "is_api": False,
    "base_tune_id": 1504944,
    "token": "ohwx",
    "args": "preset=flux-lora-fast learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4",
    "cost": None,
    "expires_at": "2025-01-04T10:43:45.717Z",
    "emailed_notice": False,
    "public_at": None,
    "face_crop": True,
    "checkpoint_deleted": False,
    "checkpoint_deleted_at": None,
    "failed_at": None,
    "model_type": "lora",
    "sha256": None,
    "model_url": None,
    "description_url": None,
    "cost_mc": 202500,
    "training_face_correct": False,
    "eta": "2024-10-06T10:36:42.918Z",
    "base_pack_id": 314,
    "characteristics": None,
    "prompts_callback": None,
    "orig_images": [
      "https://sdbooth2-production.s3.amazonaws.com/0qdf80clpct9e8qevm2urdjt1twh",
      "https://sdbooth2-production.s3.amazonaws.com/ofwcei15szoi6ykuxerov9h8a9oi",
      "https://sdbooth2-production.s3.amazonaws.com/v8t4yg7bvt9fktx0npjcquu136xk",
      "https://sdbooth2-production.s3.amazonaws.com/8tdh3a3zbttuz8f9u34quvhezpbr",
      "https://sdbooth2-production.s3.amazonaws.com/ams08x39nwnxqmxvlxi10gp6un1x",
      "https://sdbooth2-production.s3.amazonaws.com/5k3lmr24335ul00ld1221ftlou05",
      "https://sdbooth2-production.s3.amazonaws.com/1u5tq6cusx3l7s89kv1y3lnlkz4g",
      "https://sdbooth2-production.s3.amazonaws.com/98dngribbj067hshk2callk706b8",
      "https://sdbooth2-production.s3.amazonaws.com/brx7kfj283ufk6hui3p20dxrhe4t",
      "https://sdbooth2-production.s3.amazonaws.com/o1m1ykammsid2w4vep67npnmrulz",
      "https://sdbooth2-production.s3.amazonaws.com/ecdzam1rsf7t25hrsesdeqygut7e",
      "https://sdbooth2-production.s3.amazonaws.com/7o0b0jiwjg4ioa8dxfvgbvh3gxmj",
      "https://sdbooth2-production.s3.amazonaws.com/lbecdf6dhcq0ijfpcyjyctxk4hv2",
      "https://sdbooth2-production.s3.amazonaws.com/kzm3agr29i4ylftry59g3jb9fd9k",
      "https://sdbooth2-production.s3.amazonaws.com/ne8t4vfhu34lhtjzcylja052uiw7"
    ],
    "file_names": [
      {
        "filename": "ohwx (13).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/0qdf80clpct9e8qevm2urdjt1twh"
      },
      {
        "filename": "ohwx (12).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/ofwcei15szoi6ykuxerov9h8a9oi"
      },
      {
        "filename": "ohwx (8).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/v8t4yg7bvt9fktx0npjcquu136xk"
      },
      {
        "filename": "ohwx (11).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/8tdh3a3zbttuz8f9u34quvhezpbr"
      },
      {
        "filename": "ohwx (5).png",
        "url": "https://sdbooth2-production.s3.amazonaws.com/ams08x39nwnxqmxvlxi10gp6un1x"
      },
      {
        "filename": "ohwx (15).JPG",
        "url": "https://sdbooth2-production.s3.amazonaws.com/5k3lmr24335ul00ld1221ftlou05"
      },
      {
        "filename": "ohwx (7).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/1u5tq6cusx3l7s89kv1y3lnlkz4g"
      },
      {
        "filename": "ohwx (6).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/98dngribbj067hshk2callk706b8"
      },
      {
        "filename": "ohwx (2).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/brx7kfj283ufk6hui3p20dxrhe4t"
      },
      {
        "filename": "ohwx (4).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/o1m1ykammsid2w4vep67npnmrulz"
      },
      {
        "filename": "ohwx (14).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/ecdzam1rsf7t25hrsesdeqygut7e"
      },
      {
        "filename": "ohwx (1).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/7o0b0jiwjg4ioa8dxfvgbvh3gxmj"
      },
      {
        "filename": "ohwx (3).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/lbecdf6dhcq0ijfpcyjyctxk4hv2"
      },
      {
        "filename": "ohwx (9).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/kzm3agr29i4ylftry59g3jb9fd9k"
      },
      {
        "filename": "ohwx (10).jpg",
        "url": "https://sdbooth2-production.s3.amazonaws.com/ne8t4vfhu34lhtjzcylja052uiw7"
      }
    ],
    "resolution": None,
    "user": {
      "create_ckpt": False,
      "backend_version": None
    },
    "prompts": []
  })

JOB_STR_5 = json.dumps({"id":2821958,"name":"woman","created_at":"2025-06-22T19:42:20.455Z","updated_at":"2025-06-22T19:47:04.073Z","user_id":123103,"trained_at":"2025-06-22T19:47:04.072Z","started_training_at":"2025-06-22T19:42:30.017Z","steps":300,"title":"moranashual@gmail.com-1750621333516","branch":"flux1","callback":None,"process_ip":"akash-22064456-5","trials":1,"num_prompts":0,"is_api":True,"base_tune_id":1504944,"token":"ohwx","args":"preset=flux-lora-portrait learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial flux_lora_target=portrait segmentation=1 only_face=true","cost":None,"expires_at":"2025-07-22T19:47:04.072Z","emailed_notice":False,"public_at":None,"face_crop":True,"checkpoint_deleted":False,"checkpoint_deleted_at":None,"failed_at":None,"model_type":"lora","sha256":None,"model_url":None,"description_url":None,"cost_mc":150000,"training_face_correct":False,"eta":"2025-06-22T19:52:28.025Z","base_pack_id":1592,"characteristics":{"name":"woman","gender":"woman","age":"30 yo","ethnicity":"caucasian","glasses":"","eye_color":"brown eyes","hair_color":"black hair","hair_length":"medium hair","hair_style":"curly hair","facial_hair":"","is_bald":"","blurry":"false","funny_face":"false","wearing_sunglasses":"false","wearing_hat":"false","includes_multiple_people":"false","full_body_image_or_longshot":"false","selfie":"false"},"prompts_callback":"https://ai.danad.co.il/api/portrait/webhook","auto_extend":False,"orig_images":["https://mp.astria.ai/1np66zpo4cda4q0uwiwh4gtcyx7m","https://mp.astria.ai/ngy4w7ol8xyuwkdcvn9q0b7smtsg","https://mp.astria.ai/j32k5mq3ai7wyaulr0nehc0srext","https://mp.astria.ai/zis8yhylapea00966d8jcl247gir"],"file_names":[{"filename":"image_1.jpg","url":"https://sdbooth2-production.s3.amazonaws.com/1np66zpo4cda4q0uwiwh4gtcyx7m"},{"filename":"image_2.jpg","url":"https://sdbooth2-production.s3.amazonaws.com/ngy4w7ol8xyuwkdcvn9q0b7smtsg"},{"filename":"image_3.jpg","url":"https://sdbooth2-production.s3.amazonaws.com/j32k5mq3ai7wyaulr0nehc0srext"},{"filename":"image_4.jpg","url":"https://sdbooth2-production.s3.amazonaws.com/zis8yhylapea00966d8jcl247gir"}],"resolution":None,"user":{"create_ckpt":False,"backend_version":None},"prompts":[]})

JOB_STR_HAIR = json.dumps({
    "id": 2821958,
    "name": "woman",
    "created_at": "2025-06-22T19:42:20.455Z",
    "updated_at": "2025-06-22T19:47:04.073Z",
    "user_id": 123103,
    "trained_at": "2025-06-22T19:47:04.072Z",
    "started_training_at": "2025-06-22T19:42:30.017Z",
    "steps": 300,
    "title": "moranashual@gmail.com-1750621333516",
    "branch": "flux1",
    "callback": None,
    "process_ip": "akash-22064456-5",
    "trials": 1,
    "num_prompts": 0,
    "is_api": True,
    "base_tune_id": 1504944,
    "token": "ohwx",
    "args": "preset=flux-lora-portrait learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial flux_lora_target=portrait segmentation=1 use_bisenet=1 crop_expansion_factor=0.3 only_face=true",
    "cost": None,
    "expires_at": "2025-07-22T19:47:04.072Z",
    "emailed_notice": False,
    "public_at": None,
    "face_crop": True,
    "checkpoint_deleted": False,
    "checkpoint_deleted_at": None,
    "failed_at": None,
    "model_type": "lora",
    "sha256": None,
    "model_url": None,
    "description_url": None,
    "cost_mc": 150000,
    "training_face_correct": False,
    "eta": "2025-06-22T19:52:28.025Z",
    "base_pack_id": 1592,
    "characteristics": {
        "name": "woman",
        "gender": "woman",
        "age": "30 yo",
        "ethnicity": "caucasian",
        "glasses": "",
        "eye_color": "brown eyes",
        "hair_color": "black hair",
        "hair_length": "medium hair",
        "hair_style": "curly hair",
        "facial_hair": "",
        "is_bald": "",
        "blurry": "false",
        "funny_face": "false",
        "wearing_sunglasses": "false",
        "wearing_hat": "false",
        "includes_multiple_people": "false",
        "full_body_image_or_longshot": "false",
        "selfie": "false",
    },
    "prompts_callback": "https://ai.danad.co.il/api/portrait/webhook",
    "auto_extend": False,
    "orig_images": [
        "https://mp.astria.ai/1np66zpo4cda4q0uwiwh4gtcyx7m",
        "https://mp.astria.ai/ngy4w7ol8xyuwkdcvn9q0b7smtsg",
        "https://mp.astria.ai/j32k5mq3ai7wyaulr0nehc0srext",
        "https://mp.astria.ai/zis8yhylapea00966d8jcl247gir",
    ],
    "file_names": [
        {
            "filename": "image_1.jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/1np66zpo4cda4q0uwiwh4gtcyx7m",
        },
        {
            "filename": "image_2.jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/ngy4w7ol8xyuwkdcvn9q0b7smtsg",
        },
        {
            "filename": "image_3.jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/j32k5mq3ai7wyaulr0nehc0srext",
        },
        {
            "filename": "image_4.jpg",
            "url": "https://sdbooth2-production.s3.amazonaws.com/zis8yhylapea00966d8jcl247gir",
        },
    ],
    "resolution": None,
    "user": {
        "create_ckpt": False,
        "backend_version": None,
    },
    "prompts": [],
})