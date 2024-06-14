
Simple cmd-line league system for bot contests.

Install the latest version with `pip install psyleague --upgrade`

AFAIK, it requires python 3.8 or newer.

You can see the latest changes in the [changelog.txt](https://github.com/FakePsyho/psyleague/blob/main/changelog.txt). 

**Note: if you encounter any bugs/problems, feel free to contact me on twitter/discord and tell me about the issue.**

## Main Features
- League with simple automatic matchmaking system: start the league server, add bots, look at the results table
- Add/remove/modify bots without restarting the server
- All data is stored in a human-readable format
- Easy way of adding metadata to your games and the display those stats on the leaderboard; it also allows for rating recalculation on a specific subset of games
- Should work with all programming languages and all possible platforms
- [Coming in 0.5.0?] Support for different ranking models


## Quick Setup Guide
- Install `psyleague` (via `pip install psyleague --upgrade`)
- Create a play_game script that is invoked every time psyleague wants to run a new game. This is usually either a wrapper on a provided referee or your own game simulator. Please see [this example](#codingame-play_game-script) for explanation about the JSON format.
- Run `psyleague config` in your contest directory to create a new config file
- In `psyleague.cfg` you have to modify `cmd_bot_setup` and `cmd_play_game`. `cmd_bot_setup` is executed immediately when you add a new bot. `cmd_play_game` is executed when `psyleague` wants to play a single game. %DIR% -> `dir_bots` (from config), %NAME% -> `BOT_NAME`, %SRC% -> `SOURCE` (or `BOT_NAME` if `SOURCE` was not provided), %P1% & %P2% -> `BOT_NAME` of the player 1 & player 2 bots.
- If your game has more than 2 players, you should change `n_players` in the config.
- Run `psyleague run` in a terminal - this is the "server" part that automatically plays games. In order to kill it, use keyboard interrupt (Ctrl+C).
- In a different terminal, start adding bots by running `psyleague bot add BOT_NAME -s SOURCE` to add a new bot to the league. As soon as you have 2 bots added, `psyleague run` will start playing games.
- Run `psyleague show` to see the current leaderboard
- Remember to update `n_workers` (in the config or via `run --workers n_workers`) in order to play more games simultaneously 


## CodinGame play_game script
Your play_game script should print a valid JSON that contains 4 fields. P refers to the number of players.
- `ranks`: a list of length P, that describes the final placements of every player; it follows the same format as [trueskill](https://trueskill.org/) library
- `errors`: a list of length P that describes if there was an error (timeout, crash, etc.) for a particular player
- `test_data`: an object (dictionary) that contains metadata related to this particular test/seed.
- `player_data`: a list of length P that contains metadata related to each of the players.

The following python code should work with most of the CodinGame contests assuming `referee.jar` contains the referee. If your referee works with brutaltester it also works with psyleague.
```
import sys, subprocess, random, json, tempfile, os
if __name__ == '__main__':
    f, log_file = tempfile.mkstemp(prefix='log_')
    os.close(f)
    seed = random.randrange(0, 2**31)
    n_players = len(sys.argv) - 1
    cmd = 'java -jar referee.jar' + ''.join([f' -p{i} "{sys.argv[i]}"' for i in range(1, n_players+1)]) + f' -d seed={seed} -l "{log_file}"'
    task = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(log_file, 'r') as f:
        json_log = json.load(f)
    os.remove(log_file)
    p_scores = [int(json_log['scores'][str(i)]) for i in range(n_players)]
    rv = {}
    rv['ranks'] = [sum([int(p_score < p2_score) for p2_score in p_scores]) for p_score in p_scores] # assumes higher score is better
    rv['errors'] = [int(p_score < 0) for p_score in p_scores] # assumes negative score means error
    rv['test_data'] = {'seed': seed}
    rv['player_data'] = [{} for _ in range(n_players)]
    for player, key in enumerate([str(i) for i in range(n_players)]):
        for data in json_log['errors'][key]:
            if not data: continue
            for line in [line.strip() for line in data.split('\n')]:
                vs = line.split(' ')
                if len(vs) < 4: continue
                if vs[0] == '[TDATA]':  rv['test_data'][vs[1]] = vs[3]
                if vs[0] == '[PDATA]':  rv['player_data'][player][vs[1]] = vs[3]
                if vs[0] == '[PDATA+]': rv['player_data'][player][vs[1]] = rv['player_data'][player].get(vs[1], 0.0) + float(vs[3])            
    print(json.dumps(rv))
```

In order to add metadata to test_data, just print `[TDATA] key = value` to stderr in your bot. Similarly, print `[PDATA] key = value` for adding metadata to player_data. Use `[PDATA+] key = value` if you want to store the sum of all of the values instead only the last one.


## Debugging
- Most of the potential errors come from either having incorrect referee or a mistake in your play_game script. It's recommended to run your `cmd_play_game` manually and see if it correctly prints out JSON string to stdout.
- Set `n_workers` to 1 and run server with verbose turned on: `psyleague run --verbose` to see if your `cmd_play_game` script is called correctly. 

	
## Ranking Models
- trueskill
	-  this is the same model that CodinGame uses, except here you have the ability to reduce `tau` so that the ranking can stabilize after a while. Unless you're running 10K+ games per bot, there's probably no reason to reduce `tau` even more.

   
## MatchMaking
Matchmaking model is very simple:
- If there are any bots with less than `mm_min_matches` games and we have "rolled" below `mm_min_matches_preference`: play a game between a bot with not enough games and a random bot
- Otherwise: play a game between two random bots


## Scoreboard
TBD


## Other Details
- **`psyleague` is not going to be backward compatible. Every new version might break the format of any of the files and/or config.**
- **Most of the referees for CodinGame detect if your bot goes above allowed time for each turn. If you spawn too many workers your bots will start timing out randomly.**
- Config file is read only once at the startup. If you have updated config file, you have to restart `psyleague run` to reflect the changes
- You can modify `psyleague.db` to make direct changes to the bots/stats, but don't do that while `psyleague run` is running. In order to reset everything, it's enough to delete `psyleague.db` & `psyleague.games`.
- If you want to see the list of planned changes, see [the top of the source file](https://github.com/FakePsyho/psyleague/blob/main/psyleague/psyleague.py)

