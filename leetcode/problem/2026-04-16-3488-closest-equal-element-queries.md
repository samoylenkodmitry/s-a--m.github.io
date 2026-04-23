---
layout: leetcode-entry
title: "3488. Closest Equal Element Queries"
permalink: "/leetcode/problem/2026-04-16-3488-closest-equal-element-queries/"
leetcode_ui: true
entry_slug: "2026-04-16-3488-closest-equal-element-queries"
---

[3488. Closest Equal Element Queries](https://leetcode.com/problems/closest-equal-element-queries/solutions/7934760/kotlin-rust-by-samoylenkodmitry-smlt/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16042026-3488-closest-equal-element?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/QTn2NIJ70-w)

![16.04.2026.webp](/assets/leetcode_daily_images/16.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1330

#### Problem TLDR

Shortest distances to the same values #medium

#### Intuition

```j
    // [1,3,1,4,1,3,2], queries = [0,3,5]
    //  *   *   *
    // brute force is n^2
    //  0   2   4                  *
    //  for each find the closest to the right and to the left
    //  to the left
    //  ?   2   2    (and update first to (first+size-i)
    //  to the right
    //  2   2   ?    (and update last to (size-last+i))
    //
```
In a 2*n forward pass updated the distance to the previous value and to the current value.

#### Approach

* HashMap's put in Kotlin and insert in Rust returns the previous value

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 104ms
    fun solveQueries(n: IntArray, q: IntArray) = HashMap<Int,Int>().run {
        val z = n.size; val d = IntArray(z) { z }
        for (i in 0..<2*z) put(n[i%z], i)?.let { p ->
            d[i%z] = min(d[i%z], i-p); d[p%z] = min(d[p%z], i-p)
        }
        q.map { d[it].takeIf { it < z } ?: -1 }
    }
```
```rust
// 38ms
    pub fn solve_queries(n: Vec<i32>, q: Vec<i32>) -> Vec<i32> {
        let z = n.len(); let (mut p, mut d) = (HashMap::new(), vec![z;z]);
        for i in 0..2*z { if let Some(p) = p.insert(n[i%z], i) {
            d[i%z] = d[i%z].min(i-p); d[p%z] = d[p%z].min(i-p)
        }}
        q.iter().map(|&i|if d[i as usize]<z {d[i as usize] as i32}else{-1}).collect()
    }
```

