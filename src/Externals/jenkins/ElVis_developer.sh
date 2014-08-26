#!/bin/bash

me=`basename $0`

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo
  echo "Execute with $me username email [dir]"
  echo "where 'username' is your username on github," 
  echo "email is your prefered email address,"
  echo "and [dir] is an optional directory name where ElVis will be cloned"
  echo
  exit 1
fi

USERNAME=$1
EMAIL=$2
BRANCH=$USERNAME/develop
REPO_PATH=https://github.com/SCI-ElVis/ElVis.git
REPO=ssh://$REPO_PATH
REMOTE=SCI-ElVis

if [ $# == 3 ]; then
  ELVISDIR=$3
else
  ELVISDIR=.
fi

git_error()
{
  echo
  echo "git returned an error code. Aborting."
  echo
  exit 1
}


#Check if in a git repo, then check for modified files if so
if (git status --porcelain > /dev/null 2>&1); then
  while read status filename; do
    if [ $status == "M" ]; then
      git status
      echo
      echo "Please commit your changes before runnig $me."
      echo
      exit 1
    fi
  done < <(git status --porcelain)

  #Check for the correct remote and change the name to REMOTE
  let remote_found=0
  for remote in `git remote show`; do
    if [[ "`git config remote.$remote.url`" == *"$REPO_PATH"* ]]; then
       if [ ! $remote == "$REMOTE" ]; then
         echo "Renaming remote $remote to $REMOTE"
         git remote rename $remote $REMOTE || git_error
       fi
       let remote_found=1
    fi
  done
  if (( ! remote_found )); then
    echo
    echo "Could not find a remote with URL:$REPO"
    echo "This does not appear to be an ElVis git repo. Aborting."
    echo
    exit 1
  fi

  #Just make sure everything is up to date
  git fetch || git_error

  if (git branch | grep "$BRANCH"$ --quiet); then
    #The developers branch already exists, so just check it out
    echo "Checking out $BRANCH"
    git checkout $BRANCH || git_error
  elif (git ls-remote --heads $REPO | grep $BRANCH --quiet); then
    #The branch exists in the reop, check it out
    echo "Checking out $BRANCH"
    git checkout -b $BRANCH $REMOTE/$BRANCH || git_error
  else
    echo "Checking out develop branch"
    #Switch to the develop branch, and create it if needed
    if (git branch | grep " develop"$ --quiet); then
      git checkout develop || git_error
    else
      git checkout -b develop $REMOTE/develop || git_error
    fi
    #Now create BANCH so it branches of from develop
    echo "Checking out $BRANCH"
    git checkout -b $BRANCH || git_error
    git push $REMOTE $BRANCH || git_error
  fi
else
  #We are not in a git repo, so it needs to be cloned into ELVISDIR

  if (git ls-remote --heads $REPO | grep $BRANCH --quiet); then
    #The branch already exists, so just check it out
    echo "Cloning existing branch $BRANCH"
    git clone -o $REMOTE -b $BRANCH $REPO $ELVISDIR || git_error
    cd $ELVISDIR
  else
    echo "Cloning and creating new branch $BRANCH"
    git clone -o $REMOTE -b develop $REPO $ELVISDIR || git_error
    cd $ELVISDIR
    git checkout -b $BRANCH || git_error
  fi
fi

git config branch.$BRANCH.remote $REMOTE || git_error
git config --replace-all branch.$BRANCH.merge refs/heads/develop || git_error
git config --add branch.$BRANCH.merge refs/heads/$BRANCH || git_error
git config alias.update "!git branch | grep \"* $BRANCH\"$ && git fetch -p $REMOTE && git merge $REMOTE/$BRANCH && git merge $REMOTE/develop || git pull" || git_error
git config core.eol lf || git_error

for branch in master apprentice develop
do
  if (git branch | grep " $branch"$ --quiet); then
    echo "Merging local $branch into $BRANCH and removing local branch $branch"
    #Merge the master/develop branch into BRANCH just in case they had changes. 
    git merge $branch || git_error
    #Can't push to master/develop, so no point in having them checked out as a developer
    git branch -D $branch || git_error
  fi
done

#Add the email address to the list of developer email addresses
echo
echo "Adding email address '$EMAIL' to DeveloperEmailAddresses and git config user.email"
echo
git config user.email $EMAIL || git_error
EmailFile=$(git rev-parse --show-toplevel)/DeveloperEmailAddresses
grep --quiet -i $EMAIL $EmailFile || echo "`cat $EmailFile`, $EMAIL" > $EmailFile

if( ! git diff --exit-code --quiet ); then
  git add $EmailFile || git_error
  git commit -m "Added $EMAIL to DeveloperEmailAddresses" || git_error
fi

echo
echo "Please remember to use the alias 'git update' instead of 'git pull'"
echo
