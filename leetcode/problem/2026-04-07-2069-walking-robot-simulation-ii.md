---
layout: leetcode-entry
title: "2069. Walking Robot Simulation II"
permalink: "/leetcode/problem/2026-04-07-2069-walking-robot-simulation-ii/"
leetcode_ui: true
entry_slug: "2026-04-07-2069-walking-robot-simulation-ii"
---

[2069. Walking Robot Simulation II](https://leetcode.com/problems/walking-robot-simulation-ii/solutions/7806956/kotlin-rust-by-samoylenkodmitry-wu47/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07042026-2069-walking-robot-simulation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZzIQw_xsrkI)

![07.04.2026.webp](/assets/leetcode_daily_images/07.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1321

#### Problem TLDR

Implement perimeter-robot #medium

#### Intuition

```j
// 15 minute: so this TLE
//
// does robot always goes on perimeter? -- yes
//
// make it linear
//
// xxxxxyyyxxxxxyyyxxxxxyyyxxxxxyyy
//  w    h   w   h 0
//
// each rotation is extra step? - no
```

1. its only the perimeter
2. w-1, h-1
3. top-down south

#### Approach

* extract separate method

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 146ms
class Robot(a: Int, b: Int, var p: Int = 0) {
    val w=a-1; val h=b-1; val wh=w+w+h+h
    val i get() = when (val m = p % wh) {
        in 0..w -> listOf(m, 0) to if(m==0 && p>0) "South" else "East"
        in 0..w+h -> listOf(w, m-w) to "North"
        in 0..wh-h -> listOf(wh-h-m, h) to "West"
        else -> listOf(0, wh-m) to "South"
    }
    fun step(n: Int) { p += n }; fun getPos() = i.first; fun getDir() = i.second
}
```
```rust
// 16ms
struct Robot(i32, i32, i32, i32); impl Robot {
    fn new(a: i32, b: i32) -> Self { Self(a-1, b-1, 2*a+2*b-4, 0) }
    fn step(&mut self, n: i32) { self.3 += n }
    fn i(&self) -> (Vec<i32>, &'static str) {
        let (w, h, t, p, m) = (self.0, self.1, self.2, self.3, self.3 % self.2);
        if m <= w { (vec![m, 0], if p>0 && m==0 {"South"} else {"East"}) }
        else if m <= w+h { (vec![w, m-w], "North") }
        else if m <= t-h { (vec![t-h-m, h], "West") }
        else { (vec![0, t-m], "South") }
    }
    fn get_pos(&self) -> Vec<i32> { self.i().0 }
    fn get_dir(&self) -> String { self.i().1.into() }
}
```

