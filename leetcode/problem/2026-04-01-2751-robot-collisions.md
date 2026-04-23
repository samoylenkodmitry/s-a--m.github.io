---
layout: leetcode-entry
title: "2751. Robot Collisions"
permalink: "/leetcode/problem/2026-04-01-2751-robot-collisions/"
leetcode_ui: true
entry_slug: "2026-04-01-2751-robot-collisions"
---

[2751. Robot Collisions](https://open.substack.com/pub/dmitriisamoilenko/p/01042026-2751-robot-collisions?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) hard
[youtube](https://youtu.be/PYAQQAtjOW0)

![01.04.2026.webp](/assets/leetcode_daily_images/01.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1315

#### Problem TLDR

Collide LR robots #hard #stack

#### Intuition

```j
    // 1. sort by position
    // 2. simulation is too big, need a O(n) algorithm
    // 3. R R L R L
    //    2 2 1 1 3
    //    2 1 0
    //    2 1 0 0 2
    //    2 0 0 0 1
    //    1 0 0 0 0
```

Sort. Use stack.

#### Approach

* we can re-use p array as a stack

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 70ms
    fun survivedRobotsHealths(p: IntArray, h: IntArray, d: String) = run {
        var t = -1; for (i in p.indices.sortedBy { p[it] })
            if (d[i] == 'R') p[++t] = i else while (t >= 0 && h[i] > 0) when {
                h[p[t]] > h[i] -> { h[p[t]]--; h[i] = 0 }
                h[p[t]] < h[i] -> { h[p[t--]] = 0; h[i]-- }
                else -> { h[p[t--]] = 0; h[i] = 0 }
            }
        h.filter { it > 0 }
    }
```
```rust
// 11ms
    pub fn survived_robots_healths(p: Vec<i32>, mut h: Vec<i32>, d: String) -> Vec<i32> {
        let mut s = vec![];
        for i in (0..p.len()).sorted_by_key(|&i| p[i]) {
            if d.as_bytes()[i] == b'R' { s.push(i) } else {
            while let Some(t) = s.pop() {
                if h[t] > h[i] { h[t] -= 1; h[i] = 0; s.push(t); break }
                if h[t] < h[i] { h[t] = 0; h[i] -= 1; continue }
                h[t] = 0; h[i] = 0; break
            }}}
        h.into_iter().filter(|&x| x > 0).collect()
    }
```

