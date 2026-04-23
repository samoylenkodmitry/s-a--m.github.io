---
layout: leetcode-entry
title: "2483. Minimum Penalty for a Shop"
permalink: "/leetcode/problem/2025-12-26-2483-minimum-penalty-for-a-shop/"
leetcode_ui: true
entry_slug: "2025-12-26-2483-minimum-penalty-for-a-shop"
---

[2483. Minimum Penalty for a Shop](https://leetcode.com/problems/minimum-penalty-for-a-shop/description/) medium
[blog post](https://leetcode.com/problems/minimum-penalty-for-a-shop/solutions/7440413/kotlin-rust-by-samoylenkodmitry-2ycl/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26122025-2483-minimum-penalty-for?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/iDEQQnc6tMk)

![cc090fff-c7c3-4399-aa61-414eaa4ba39f (1).webp](/assets/leetcode_daily_images/22925f1c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1216

#### Problem TLDR

Min close time by customers max volune #medium #counting

#### Intuition

Track of the customers volune running sum. YN pairs didn't change the volume.

#### Approach

* 'Y'-'N'=11
* baseline can be anything

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 19ms
    fun bestClosingTime(c: String): Int {
        var v = 1337
        return (0..c.length).maxBy { v += 2*(c[max(0,it-1)]-'N')/10-1; v }
    }
```
```rust
// 0ms
    pub fn best_closing_time(c: String) -> i32 {
        let mut v = 404;
        (0..=c.len()as i32).max_by_key(|&i|{
            v += 2 * (c.as_bytes()[0.max(i-1) as usize]==b'Y')as i32-1; (v,-i)
        }).unwrap()
    }
```

