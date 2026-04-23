---
layout: leetcode-entry
title: "874. Walking Robot Simulation"
permalink: "/leetcode/problem/2026-04-06-874-walking-robot-simulation/"
leetcode_ui: true
entry_slug: "2026-04-06-874-walking-robot-simulation"
---

[874. Walking Robot Simulation](https://leetcode.com/problems/walking-robot-simulation/solutions/7792264/kotlin-rust-by-samoylenkodmitry-f338/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06042026-874-walking-robot-simulation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/AT9HQoST73I)

![06.04.2026.webp](/assets/leetcode_daily_images/06.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1320

#### Problem TLDR

Farthest robot movement #medium #simulation

#### Intuition

Just simulate the movements, 1..9 can be iterated.

#### Approach

* to compress position into a single variable, we have to make coordinates positive

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(o)$$

#### Code

```kotlin
// 41ms
    fun robotSim(c: IntArray, o: Array<IntArray>): Int {
        val S = o.map { it[1]+32768 shl 16 or it[0]+32768 }.toSet()
        var d = 0; var p = -2147450880; val D = listOf(65536, 1, -65536, -1)
        return c.maxOf {
            if (it < 0) d = (d - 2*it - 1) % 4
            repeat(it) { if (p+D[d] !in S) p += D[d] }
            val x = (p and 65535) - 32768; val y = (p ushr 16) - 32768
            x*x + y*y
        }
    }
```
```rust
// 1ms
    pub fn robot_sim(c: Vec<i32>, o: Vec<Vec<i32>>) -> i32 {
        let s: HashSet<_> = o.iter().map(|v| v[1]+32768<<16 | v[0]+32768).collect();
        let (mut p, mut d, D) = (-2147450880,0, [65536, 1, -65536, -1]);
        c.into_iter().fold(0, |m, i| {
            if i < 0 {  d = (d + if i < -1 { 3 } else { 1 }) % 4  }
            for _ in 0..i { if !s.contains(&(p + D[d])) { p += D[d] } }
            let (x, y) = ((p & 65535) - 32768, (p as u32 >> 16) as i32 - 32768);
            m.max(x*x + y*y)
        })
    }
```

