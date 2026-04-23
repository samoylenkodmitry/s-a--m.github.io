---
layout: leetcode-entry
title: "1970. Last Day Where You Can Still Cross"
permalink: "/leetcode/problem/2025-12-31-1970-last-day-where-you-can-still-cross/"
leetcode_ui: true
entry_slug: "2025-12-31-1970-last-day-where-you-can-still-cross"
---

[1970. Last Day Where You Can Still Cross](https://leetcode.com/problems/last-day-where-you-can-still-cross/description) hard
[blog post](https://leetcode.com/problems/last-day-where-you-can-still-cross/solutions/7453066/kotlin-rust-by-samoylenkodmitry-pqwy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31122025-1970-last-day-where-you?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UBTHKxshf2M)

![8e112f71-96d8-43bf-bc42-2dc7f46a2be2 (1).webp](/assets/leetcode_daily_images/ae814ad5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1221

#### Problem TLDR

Last day top connected to bottom in 2D matrix #hard #uf

#### Intuition

Invert the problem: go from back and the list becames a "rain of the ground cells".

#### Approach

* we can store ground bits inside union-find jump array as an extra bit

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 68ms
    val u=IntArray(21000){it*2}; fun f(a:Int):Int=if(u[a]/2==a)a else f(u[a]/2).also{u[a]=it*2+u[a]%2}
    fun latestDayToCross(r: Int, c: Int, cs: Array<IntArray>) = (cs.lastIndex downTo 0).first { i ->
        val (y,x)=cs[i]; val s = 1+2*f(if(y==1)0 else if(y==r)2 else y*c+x); u[y*c+x]=s
        for ((x1,y1) in arrayOf(x-1 to y, x+1 to y, x to y-1, x to y+1))
            if (y1 in 1..r && x1 in 1..c && u[y1*c+x1]%2>0) u[f(y1*c+x1)] = s
        f(0) == f(2)
    }
```
```rust
// 14ms
    pub fn latest_day_to_cross(r: i32, c: i32, cs: Vec<Vec<i32>>) -> i32 {
        let (r, c) = (r as usize, c as usize); let mut u: Vec<_> = (0..r*c+c+1).map(|i| i * 2).collect();
        fn f(u: &mut Vec<usize>, a: usize)->usize{if u[a]/2==a{a}else{let t=f(u,u[a]/2);u[a]=t*2+u[a]%2;t}}
        (0..cs.len()).rev().find(|&i| {
            let (y, x) = (cs[i][0] as usize, cs[i][1] as usize);
            let s = 1+2*f(&mut u,if y==1 {0} else if y == r {2} else {y * c + x}); u[y * c + x] = s;
            for (dx, dy) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                if dy >= 1 && dy <= r && dx >= 1 && dx <= c && u[dy * c + dx] % 2 > 0
                    { let r = f(&mut u, dy * c + dx); u[r] = s }}
            f(&mut u, 0) == f(&mut u, 2)
        }).map(|i| i as i32).unwrap_or(-1)
    }
```

