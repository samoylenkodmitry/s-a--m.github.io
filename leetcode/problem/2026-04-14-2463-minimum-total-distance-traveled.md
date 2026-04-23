---
layout: leetcode-entry
title: "2463. Minimum Total Distance Traveled"
permalink: "/leetcode/problem/2026-04-14-2463-minimum-total-distance-traveled/"
leetcode_ui: true
entry_slug: "2026-04-14-2463-minimum-total-distance-traveled"
---

[2463. Minimum Total Distance Traveled](https://leetcode.com/problems/minimum-total-distance-traveled/solutions/7902012/kotlin-rust-by-samoylenkodmitry-r0vx/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14042026-2463-minimum-total-distance?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wYW7w8v4lbg)

![14.04.2026.webp](/assets/leetcode_daily_images/14.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1328

#### Problem TLDR

Shortest distances to travel robots to factories #hard #dp

#### Intuition

Didn't solve without the hints.

```j
    // does speed matters?
    //
    // match each robot to each factory?
    //
    // r r f f
    //
    // the dp state can be cached by robots directions in a tail + factories
    //
    // the only thing is: does the order/speed matter?
    //
    // another idea: start BFS from each robot
    //               remove factory hits as repaired
    // but: positions are 10^9, we can't simulate travel
    //
    // can travel by entire distance at once?
    // put in a next_timestamp priority queue
    //
    // 27 minute wrong answer, mine bigger, meaning not optimal
    //                                      bfs didn't work here
    // [9,11,99,101] [[10,1],[7,1],[14,1],[100,1],[96,1],[103,1]]
    // 30 minute look for hint: segments
    //
    // r f r r f
    // *****
    //
```

Consider each robot from left to right to pick or skip factory from left to right.

#### Approach

* iterative version can save some space

#### Complexity

- Time complexity:
$$O(nfl)$$

- Space complexity:
$$O(nfl)$$

#### Code

```kotlin
// 254ms
    fun minimumTotalDistance(r: List<Int>, f: Array<IntArray>): Long {
        val r = r.sorted(); val dp = HashMap<Int, Long>()
        val f = f.sortedBy { it[0] }.flatMap { f -> List(f[1]) { f[0] }}
        fun dfs(i: Int, j: Int): Long = if (i == r.size) 0L else
            if (j == f.size) Long.MAX_VALUE/2 else
            dp.getOrPut(i*10000+j) { min(abs(r[i]-f[j]) + dfs(i+1, j+1), dfs(i, j+1)) }
        return dfs(0, 0)
    }
```
```rust
// 2ms
    pub fn minimum_total_distance(r: Vec<i32>, f: Vec<Vec<i32>>) -> i64 {
        let f: Vec<_> = f.into_iter().sorted_by_key(|v|v[0])
                         .flat_map(|v|vec![v[0]; v[1] as usize]).collect();
        let mut dp = vec![0; f.len() + 1];
        for p in r.into_iter().sorted() {
            let mut prev = dp[0]; dp[0] = i64::MAX/2;
            for j in 0..f.len() {
                let t = dp[j+1]; dp[j+1] = ((p-f[j]).abs() as i64 + prev).min(dp[j]); prev = t }
        } dp[f.len()]
    }
```

