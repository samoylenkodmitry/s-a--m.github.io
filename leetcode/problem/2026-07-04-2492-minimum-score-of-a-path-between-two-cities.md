---
layout: leetcode-entry
title: "2492. Minimum Score of a Path Between Two Cities"
permalink: "/leetcode/problem/2026-07-04-2492-minimum-score-of-a-path-between-two-cities/"
leetcode_ui: true
entry_slug: "2026-07-04-2492-minimum-score-of-a-path-between-two-cities"
---

[2492. Minimum Score of a Path Between Two Cities](https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/solutions/8374992/kotlin-rust-by-samoylenkodmitry-l3zn/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04072026-2492-minimum-score-of-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Q6dW169goxk)

https://dmitrysamoylenko.com/leetcode/

![04.07.2026.webp](/assets/leetcode_daily_images/04.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1410

#### Problem TLDR

Min in connected edges

#### Intuition

Use Union-Find to find the connected group.

#### Approach

* uf path compression: u[x]=f(u[x]) is enough speed up
* don't forget to initialize with own indices u = 1..n

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun minScore(n: Int, r: Array<IntArray>): Int {
        val u = IntArray(n+1){it}
        fun f(x: Int): Int = if (x==u[x])x else {u[x]=f(u[x]);u[x]}
        for ((a,b) in r) u[f(a)] = f(b)
        return r.minOf{(a,b,d) -> if (f(a)==f(1)) d else 999999}
    }
```
```rust
    pub fn min_score(n: i32, r: Vec<Vec<i32>>) -> i32 {
        let mut u: Vec<_> = (0..=n as usize).collect();
        let mut f = |x: i32, u: &mut [usize]| {
            let mut x = x as usize; while u[x] != x { u[x] = u[u[x]]; x = u[x] }; x };
        for e in &r { let (a,b) = (f(e[0], &mut u), f(e[1], &mut u)); u[a] = b }
        r.iter().filter(|e| f(e[0], &mut u) == f(1, &mut u)).map(|e| e[2]).min().unwrap()
    }
```

