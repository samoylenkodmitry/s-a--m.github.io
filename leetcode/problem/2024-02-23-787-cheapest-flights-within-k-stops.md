---
layout: leetcode-entry
title: "787. Cheapest Flights Within K Stops"
permalink: "/leetcode/problem/2024-02-23-787-cheapest-flights-within-k-stops/"
leetcode_ui: true
entry_slug: "2024-02-23-787-cheapest-flights-within-k-stops"
---

[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/description/) medium
[blog post](https://leetcode.com/problems/cheapest-flights-within-k-stops/solutions/4770565/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23022024-787-cheapest-flights-within?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vzlJMoFQ3Pc)

![image.png](/assets/leetcode_daily_images/53a7bea5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/516

#### Problem TLDR

Cheapest travel src -> dst with at most k stops in a directed weighted graph.

#### Approach

There is a Floyd-Warshall algorithm for such problems: make `k` rounds of travel trough all the reachable edges and improve the so-far cost.

* we must make a copy of the previous step, to avoid flying more than one step in a round

#### Complexity

- Time complexity:
$$O(kne)$$, where `e` is edges

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findCheapestPrice(n: Int, flights: Array<IntArray>, src: Int, dst: Int, k: Int): Int {
      val costs = IntArray(n) { Int.MAX_VALUE / 2 }
      costs[src] = 0
      repeat(k + 1) {
        val prev = costs.clone()
        for ((f, t, c) in flights)
            costs[t] = min(costs[t], prev[f] + c)
      }
      return costs[dst].takeIf { it < Int.MAX_VALUE / 2 } ?: -1
    }

```
```rust

  pub fn find_cheapest_price(n: i32, flights: Vec<Vec<i32>>, src: i32, dst: i32, k: i32) -> i32 {
    let mut costs = vec![i32::MAX / 2 ; n as usize];
    costs[src as usize] = 0;
    for _ in 0..=k {
      let prev = costs.clone();
      for e in &flights {
        costs[e[1] as usize] = costs[e[1] as usize].min(prev[e[0] as usize] + e[2])
      }
    }
    if costs[dst as usize] < i32::MAX / 2 { costs[dst as usize] } else { -1 }
  }

```

