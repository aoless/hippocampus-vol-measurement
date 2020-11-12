echo "*********************************************************************************"
echo "*                                                                               *"
echo "*  Here is the URL that you can use to access OHIF viewer from your web browser *"
echo "*                                                                               *"
echo "*   https://$WORKSPACEID-3005.$WORKSPACEDOMAIN    *"
echo "*                                                                               *"
echo "*                                                                               *"

cd /opt/aihcnd-applications/OHIF-Viewer/
yarn config set workspaces-experimental true
yarn run dev:orthanc