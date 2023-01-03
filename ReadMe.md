
Simple cmd-line league system for bot contests.

Install the latest version with `pip install psyleague --upgrade`

AFAIK, it requires python 3.8 or newer.

You can see the latest changes in the [changelog.txt](https://github.com/FakePsyho/psyleague/blob/main/changelog.txt). 

**Note: if you encounter any bugs/problems, feel free to contact me on twitter/discord and tell me about the issue.**

## Main Features
- League with simple automatic matchmaking system: Start the league server, add bots, look at the results table
- Add/remove/modify bots without restarting the server
- All data stored in human-readable format
- Should work with all programming languages
- [Soon!] Support for different ranking models
## Quick Setup Guide
- Install `psyleague`
- Create a script that given two bots, simulates the game and prints out 4 integers on a single line: "P1_rank P2_rank P1_error P2_error". The first two are ranks of the bots, i.e. if first player wins print "0 1", if the second player wins print "1 0", in case of a draw print "0 0". The last two signify that bot crashed during the game: 0 = no error, 1 = error.
- Run `psyleague config` in your contest directory to create a new config file
- In `psyleague.cfg` you have to modify `cmd_bot_setup` and `cmd_play_game`. `cmd_bot_setup` is executed immediately when you add a new bot. `cmd_play_game` is executed when `psyleague` wants to play a single game. %DIR% -> `dir_bots` (from config), %NAME% -> `BOT_NAME`, %SRC% -> `SOURCE` (or `BOT_NAME` if `SOURCE` was not provided), %P1% & %P2% -> `BOT_NAME` of the player 1 & 2 bots.
- Run `psyleague run` in a terminal - this is the "server" part that automatically plays games. In order to kill it, use keyboard interrupt (Ctrl+C).
- In a different terminal, start adding bots by running `psyleague bot add BOT_NAME -s SOURCE` to add a new bot to the league. As soon as you have 2 bots added, `psyleague run` will start playing games.
- Run `psyleague show` to see the current leaderboard
- Remember to update `n_workers` in order to play more games simultaneously 

## Debugging
- Make sure that your script for playing games works correctly and the only thing it prints to the stdout is the list of integers on a single line. Anything printed to stderr will be ignored. For example, the following python code is going to work for most of the CodinGame contests, assuming that `referee.jar` contains the modified judge that accepts two bots:
```
import sys, subprocess, random
if __name__ == '__main__':
    cmd = f'java -jar referee.jar -p1 "{sys.argv[1]}" -p2 "{sys.argv[2]}" -d seed={random.randrange(0, 2**31)}'
    task = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output = task.stdout.decode('UTF-8').split('\n')
    p1_score = int(output[-5])
    p2_score = int(output[-4])
    print(int(p1_score < p2_score), int(p2_score < p1_score), int(p1_score < 0), int(p2_score < 0))
```
- Set `n_workers` to 1 and run server with verbose turned on: `psyleague run --verbose` to see if your `cmd_play_game` script is called correctly. 
        
## Ranking Models
- trueskill
	-  this is the same model that CodinGame uses, except here you have the ability to reduce `tau` so that the ranking can stabilize after a while. Unless you're running 10K+ games per bot, there's probably no reason to reduce `tau` even more. 
## MatchMaking
TBD
## Scoreboard
TBD
## Other Details
- **`psyleague` is not going to be backward compatible. Every new version might break the format of any of the files and/or config.**
- **Most of the referees for CodinGame detect if your bot goes above allowed time for each turn. If you spawn too many workers your bots will start timing out randomly.**
- Config file is read only once at the startup. If you have updated config file, you have to restart `psyleague run` to reflect the changes
- You can modify `psyleague.db` to make direct changes to the bots/stats, but don't do that while `psyleague run` is running. In order to reset everything, it's enough to delete `psyleague.db` & `psyleague.games`.
- If you want to see the list of planned changes, see [the top of the source file](https://github.com/FakePsyho/psyleague/blob/main/psyleague/psyleague.py)
- Currently `psyleague.games` is used only for logging, but in the future you'll have the ability to recalculate ratings under a different ranking model

