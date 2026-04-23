---
layout: leetcode-entry
title: "3418. Maximum Amount of Money Robot Can Earn"
permalink: "/leetcode/problem/2026-04-02-3418-maximum-amount-of-money-robot-can-earn/"
leetcode_ui: true
entry_slug: "2026-04-02-3418-maximum-amount-of-money-robot-can-earn"
---

[3418. Maximum Amount of Money Robot Can Earn](https://open.substack.com/pub/dmitriisamoilenko/p/02042026-3418-maximum-amount-of-money?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[youtube](https://youtu.be/X5qd8zJvxbg)

![02.04.2026.webp](/assets/leetcode_daily_images/02.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1316

#### Problem TLDR

Max sum path, 2 skips allowed #medium #dp

#### Intuition

```j
    // how to pick two optimal robbers?
    //
    // dp?
    //
    // can be greedy (min,max)? - no, we have to track all possibilites
    // and they will grow at 2^x rate
    // so should be dp
    //
```

The top-down DFS + memo is the simplest and robust choice without extra thinking required.

#### Approach

* bottom-up: for each cell store 3 best values: (zero skips left, 1 skip left, 2 skips left)

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin
// 329ms
    fun maximumAmount(c: Array<IntArray>) = HashMap<Int, Int>().run {
        fun d(y: Int, x: Int, r: Int): Int = getOrPut(y*4000+x*4+r) {
            c.getOrNull(y)?.getOrNull(x)?.let { v ->
                fun n(k: Int) = max(d(y+1, x, k), d(y, x+1, k))
                if (v < 0 && r > 0) max(v + n(r), n(r-1)) else v + n(r)
            } ?: if (y == c.size && x == c[0].size - 1) 0 else -9999999
        }
        d(0, 0, 2)
    }
```
```rust
// 7ms
    pub fn maximum_amount(c: Vec<Vec<i32>>) -> i32 {
        let mut d = vec![[-9999999;3]; c[0].len()+1]; d[1] = [0;3];
        for r in c { for x in 1..d.len() {
            let (v, p) = (r[x-1], [0,1,2].map(|i| d[x-1][i].max(d[x][i])));
            d[x] = [p[0]+v, p[0].max(p[1]+v), p[1].max(p[2]+v)]
        }} d[d.len()-1][2]
    }
```

