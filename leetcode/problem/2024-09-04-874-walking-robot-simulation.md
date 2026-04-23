---
layout: leetcode-entry
title: "874. Walking Robot Simulation"
permalink: "/leetcode/problem/2024-09-04-874-walking-robot-simulation/"
leetcode_ui: true
entry_slug: "2024-09-04-874-walking-robot-simulation"
---

[874. Walking Robot Simulation](https://leetcode.com/problems/walking-robot-simulation/description/) medium
[blog post](https://leetcode.com/problems/walking-robot-simulation/solutions/5734958/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04092024-874-walking-robot-simulation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HX22acuqLXw)
![1.webp](/assets/leetcode_daily_images/fc9f72cc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/724

#### Problem TLDR

Max distance after robot moves simulation #medium #simulation

#### Intuition

Simulate the process. There will be at most `10 * N` steps, and we must do the obstacles checks in O(1).

#### Approach

* use the HashMap of pairs, no need to convert to strings (but can use bitset arithmetic)
* let's use iterators
* instead of direction we can use rotation matrix https://en.wikipedia.org/wiki/Rotation_matrix

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(o)$$, `o` for obstacles

#### Code

```kotlin

    fun robotSim(commands: IntArray, obstacles: Array<IntArray>): Int {
        var set = obstacles.map { it[0] to it[1] }.toSet()
        var dx = 0; var dy = 1; var x = 0; var y = 0
        return commands.maxOf { c ->
            if (c < -1) dx = -dy.also { dy = dx }
            else if (c < 0) dx = dy.also { dy = -dx }
            else for (i in 1..c) if (((x + dx) to (y + dy)) !in set)
                { x += dx; y += dy }
            y * y + x * x
        }
    }

```
```rust

    pub fn robot_sim(commands: Vec<i32>, obstacles: Vec<Vec<i32>>) -> i32 {
        let set: HashSet<_> = obstacles.into_iter().map(|o| (o[0], o[1])).collect();
        let (mut dx, mut dy, mut x, mut y) = (0, 1, 0, 0);
        commands.iter().map(|&c| {
            if c < -1 { (dx, dy) = (-dy, dx) }
            else if c < 0 { (dx, dy) = (dy, -dx) }
            else { for i in 0..c { if !set.contains(&(x + dx, y + dy)) {
                x += dx; y += dy
            }}}
            x * x + y * y
        }).max().unwrap()
    }

```
```c++

    int robotSim(vector<int>& commands, vector<vector<int>>& obstacles) {
        std::unordered_set<long long> obs;
        for (const auto& o : obstacles)
            obs.insert((long long)o[0] << 32 | (unsigned int)o[1]);
        int dx[] = {0, 1, 0, -1}, dy[] = {1, 0, -1, 0}, x = 0, y = 0, di = 0, res = 0;
        for (int c : commands)
            if (c < 0) di = (di + (c == -1 ? 1 : 3)) % 4;
            else while (c-- && !obs.count((long long)x + dx[di] << 32 | (unsigned int)(y + dy[di])))
                x += dx[di], y += dy[di], res = std::max(res, x*x + y*y);
        return res;
    }

```

