---
layout: leetcode-entry
title: "2463. Minimum Total Distance Traveled"
permalink: "/leetcode/problem/2024-10-31-2463-minimum-total-distance-traveled/"
leetcode_ui: true
entry_slug: "2024-10-31-2463-minimum-total-distance-traveled"
---

[2463. Minimum Total Distance Traveled](https://leetcode.com/problems/minimum-total-distance-traveled/description/) hard
[blog post](https://leetcode.com/problems/minimum-total-distance-traveled/solutions/5989292/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31102024-2463-minimum-total-distance?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LEgNQMGFoA0)
[deep-dive](https://notebooklm.google.com/notebook/0e4499eb-50d6-413e-838e-938deb7bc08b/audio)
![1.webp](/assets/leetcode_daily_images/c8eba8b2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/786

#### Problem TLDR

Min robots[x] travel to factories[x,capacity] #hard #dynamic_programming #sorting

#### Intuition

Failed to solve without hints.
Some ideas:
* time to travel is not considered, only the capacity
* each factory can take or not take several robots
* it feels optimal to take the closest robots to the factory (so, sort by x coordinate), but in some cases it is not (so, probably need to search all possibilities)

The hint is: factory takes closes range of robots to the left, then we go to the next factory.

The DP is $$dp[f][r] = min_{j=0..capacity_f}(sum(dist_j) + dp[f-1][r-1])$$, dp[f][r] - is the optimal travel of all robots ending with `r` by all factories ending with `f`.

#### Approach

* top down DP can be easier to write, just consider the current element and take it or not, then add a cache
* take a hint after ~30 minutes
* Rust `usize` conversion of indices can shoot at a foot, better calculate in `i32` then convert
* C++ is very good for codegolf
* we only have at most `100` robots and factories

#### Complexity

- Time complexity:
$$O(kn^2)$$, k is factories capacity

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun minimumTotalDistance(robot: List<Int>, factory: Array<IntArray>): Long {
        factory.sortBy { it[0] }; val rs = robot.sorted()
        val dp = Array(factory.size + 1) { LongArray(rs.size + 1) { Long.MAX_VALUE / 2 } }
        for ((f, fac) in factory.withIndex()) for (r in 0..<rs.size) {
            var dist = 0L; dp[f + 1][r] = dp[f][r]
            for (ri in r downTo max(0, r - fac[1] + 1)) {
                dist += abs(rs[ri] - fac[0])
                dp[f + 1][r] = min(dp[f + 1][r], dist + if (ri < 1) 0L else dp[f][ri - 1])
            }
        }
        return dp[factory.size][rs.size - 1]
    }

```
```rust

    pub fn minimum_total_distance(mut robot: Vec<i32>, mut factory: Vec<Vec<i32>>) -> i64 {
        robot.sort_unstable(); factory.sort_unstable_by_key(|f| f[0]);
        let mut dp = vec![vec![i64::MAX / 2; robot.len() + 1]; factory.len() + 1];
        for f in 0..factory.len() { let fac = &factory[f]; for r in 0..robot.len() {
            let mut dist = 0; dp[f + 1][r] = dp[f][r];
            for ri in (0.max(r as i32 + 1 - fac[1]) as usize..=r).rev() {
                dist += (robot[ri] - fac[0]).abs() as i64;
                let prev = if ri < 1 { 0 } else { dp[f][ri - 1] };
                dp[f + 1][r] = dp[f + 1][r].min(dist + prev);
            }
        }}
        dp[factory.len()][robot.len() - 1]
    }

```
```c++

    long long minimumTotalDistance(vector<int>& r, vector<vector<int>>& f) {
        sort(begin(r), end(r)); sort(begin(f), end(f));
        static long long d[101][101]; fill_n(&d[0][0], 10201, 1e18);
        for (int i = 0; i < f.size(); ++i)
            for (int j = 0; d[i + 1][j] = d[i][j], j < r.size(); ++j)
                for (long long s = 0, k = j; k >= max(0, j - f[i][1] + 1); --k)
                    d[i + 1][j] = min(d[i + 1][j], (s += abs(r[k] - f[i][0])) + (k ? d[i][k - 1] : 0));
        return d[f.size()][r.size() - 1];
    }

```

