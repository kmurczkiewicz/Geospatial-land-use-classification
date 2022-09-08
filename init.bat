@ECHO OFF
SETLOCAL

SET script_path=%~dp0
SET project_path=%script_path:~0,-1%

md %project_path%\artefacts\dataset
md %project_path%\artefacts\model_checkpoint
md %project_path%\artefacts\models_pb
md %project_path%\artefacts\sat_images
md %project_path%\artefacts\sat_images\tiles
md %project_path%\artefacts\sat_images\tiles_map

echo "Local environment initialized."

ENDLOCAL