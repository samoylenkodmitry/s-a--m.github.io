---
layout: leetcode-entry
title: "1833. Maximum Ice Cream Bars"
permalink: "/leetcode/problem/2026-06-21-1833-maximum-ice-cream-bars/"
leetcode_ui: true
entry_slug: "2026-06-21-1833-maximum-ice-cream-bars"
---

[1833. Maximum Ice Cream Bars](https://leetcode.com/problems/maximum-ice-cream-bars/solutions/8348870/kotlin-rust-by-samoylenkodmitry-2xz1/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21062026-1833-maximum-ice-cream-bars?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XH5RbKScx04)

https://dmitrysamoylenko.com/leetcode/

![21.06.2026.webp](/assets/leetcode_daily_images/21.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1397

#### Problem TLDR

Max count by given coins

#### Intuition

Maintain the frequency array. Calculate count by dividing total couns by price. Subtract count * price.

#### Approach

* pre-calculate the max

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maxIceCream(c: IntArray, k: Int) = run {
        val s = c.groupBy{it}; var k = k
        (1..c.max()).sumOf {i->min(s[i]?.size?:0, k/i).also{k -= it*i}}
    }
```
```rust
    pub fn max_ice_cream(c: Vec<i32>, mut k: i32) -> i32 {
        let m=*c.iter().max().unwrap();let mut s=vec![0;m as usize+1];
        for x in c{s[x as usize]+=1}
        (1..=m).map(|i|{let r=s[i as usize].min(k/i);k-=r*i;r}).sum()
    }
```

