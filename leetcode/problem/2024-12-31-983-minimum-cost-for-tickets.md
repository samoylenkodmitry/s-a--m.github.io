---
layout: leetcode-entry
title: "983. Minimum Cost For Tickets"
permalink: "/leetcode/problem/2024-12-31-983-minimum-cost-for-tickets/"
leetcode_ui: true
entry_slug: "2024-12-31-983-minimum-cost-for-tickets"
---

[983. Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/description/) medium
[blog post](https://leetcode.com/problems/minimum-cost-for-tickets/solutions/6208888/kotlin-rust-by-samoylenkodmitry-e4pb/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31122024-983-minimum-cost-for-tickets?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/gKUzRxyO-1Q)
[deep-dive](https://notebooklm.google.com/notebook/c2d9fd33-b10b-40d0-a501-531fd5c0731b/audio)
![1.webp](/assets/leetcode_daily_images/7776111f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/849

#### Problem TLDR

Min sum buying 1,7,30-days tickets to travel all days #medium #dymanic_programming

#### Intuition

Observing the data:

```j

    // 1,2,3,4,5,6,7,8,9,10,29,30,31      2 7 15
    // *           .                 +2
    //   *         .                 +2 4
    //     *       .                 +2 6
    //       *     .                 +2 8 vs 7, take 7,
    //             .                  from = max(1, 4-7)
    //         . . .                  to = from+7
    //               *               +2 9
    //                 *             +2 11
    //                   *           +2 13

```
* we can retrospectively switch previous ticket from `1`-day to `7` day or `30` days if it is a less expensive (this is a cleverer solution and requires a clever implementation, so initially I've dropped this idea)
* the tail (or the head) is independent and can be calculated separately, meaning, we can do a full Depth-First search and cache the result

#### Approach

* top-down DFS is easier to reason about: do choices, choose the best, then add the caching
* then rewrite to the bottom-up, reverse an iteration order for the CPU cache speed
* the idea of retrospectively replacing the 1-day ticket for 7 or 30-days can be written with queues of 7-day and 30-days ticket results: pop expired from the front, add the current to the tail, best result are at the front (c++ solution)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, O(1) for queues, as only the last 30 days are considered

#### Code

```kotlin

    val dp = HashMap<Int, Int>()
    fun mincostTickets(days: IntArray, costs: IntArray, start: Int = 0): Int =
        if (start < days.size) dp.getOrPut(start) {
            var i = start
            costs.zip(listOf(1, 7, 30)).minOf { (c, d) ->
                while (i < days.size && days[i] - days[start] < d) ++i
                c + mincostTickets(days, costs, i)
            }
        } else 0

```
```rust

    pub fn mincost_tickets(days: Vec<i32>, costs: Vec<i32>) -> i32 {
        let mut dp = vec![i32::MAX; days.len() + 1]; dp[0] = 0;
        for start in 0..days.len() {
            let mut i = start;
            for (c, d) in costs.iter().zip([1, 7, 30]) {
                while i < days.len() && days[i] - days[start] < d { i += 1 }
                dp[i] = dp[i].min(dp[start] + c)
            }
        }; dp[days.len()]
    }

```
```c++

    int mincostTickets(vector<int>& days, vector<int>& costs) {
        queue<pair<int, int>> last7, last30; int res = 0;
        for (auto d: days) {
            while (last7.size() && last7.front().first + 7 <= d) last7.pop();
            while (last30.size() && last30.front().first + 30 <= d) last30.pop();
            last7.push({d, res + costs[1]});
            last30.push({d, res + costs[2]});
            res = min({res + costs[0], last7.front().second, last30.front().second});
        } return res;
    }

```

