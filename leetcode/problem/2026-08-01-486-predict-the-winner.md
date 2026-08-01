---
layout: leetcode-entry
title: "486. Predict the Winner"
permalink: "/leetcode/problem/2026-08-01-486-predict-the-winner/"
leetcode_ui: true
entry_slug: "2026-08-01-486-predict-the-winner"
---

[486. Predict the Winner](https://leetcode.com/problems/predict-the-winner/solutions/8434255/kotlin-rust-by-samoylenkodmitry-f515/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01082026-486-predict-the-winner?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/psovw3BwIt8)

https://dmitrysamoylenko.com/leetcode/

![01.08.2026.webp](/assets/leetcode_daily_images/01.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1438

#### Problem TLDR

Optimal choices to win

#### Intuition

DFS: take the left or take the right; add to your sum, your tail is the total sum - opponent sum

#### Approach

* problem is small, no need for dp
* even size: player 1 always wins

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
    fun predictTheWinner(n: IntArray): Boolean {
        fun d(from:Int, to:Int): Int = if (from==to) n[from]
        else max(n[from]-d(from+1,to), n[to]-d(from,to-1))
        return n.size%2<1 || d(0,n.size-1)>=0
    }
```
```rust
    pub fn predict_the_winner(n: Vec<i32>) -> bool {
        let mut d = [0;21];
        for i in 0..n.len() { for j in (0..=i).rev() {
            d[j]=(n[i]-d[j]).max(n[j]-d[j+1])
        }}; d[0] >= 0
    }
```

