FROM fairembodied/habitat-challenge:testing_2021_habitat_base_docker

RUN /bin/bash -c ". activate habitat; conda install pytorch==1.7.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch; conda install tensorboard; pip install ifcfg"

ADD il_agent.py agent.py
ADD submission.sh submission.sh
ADD configs/challenge_objectnav2021.local.rgbd.yaml /challenge_objectnav2021.local.rgbd.yaml
ENV AGENT_EVALUATION_TYPE remote

# Add checkpoints and custom configs
ADD ckpts/model_9.ckpt ckpt/model.ckpt

ADD configs/il_objectnav.yaml configs/il_objectnav.yaml

# Add src folder for models
ADD src/ src/


ENV AGENT_CONFIG_FILE "configs/il_objectnav.yaml"
ENV TRACK_CONFIG_FILE "/challenge_objectnav2021.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path ckpt/model.ckpt --config-path $AGENT_CONFIG_FILE"]
