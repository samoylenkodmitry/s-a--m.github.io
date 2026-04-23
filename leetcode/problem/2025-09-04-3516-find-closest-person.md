---
layout: leetcode-entry
title: "3516. Find Closest Person"
permalink: "/leetcode/problem/2025-09-04-3516-find-closest-person/"
leetcode_ui: true
entry_slug: "2025-09-04-3516-find-closest-person"
---

[3516. Find Closest Person](https://leetcode.com/problems/find-closest-person/description/) easy
[blog post](https://leetcode.com/problems/find-closest-person/solutions/7154114/kotlin-rust-by-samoylenkodmitry-8try/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04092025-3516-find-closest-person?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ed9Bm7Q0YxM)

![1.webp](/assets/leetcode_daily_images/536f2410.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1102

#### Problem TLDR

Compare two distances #easy

#### Intuition

Distance is `abs(z - x or y)`

#### Approach

* use `when`

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 14ms
    fun findClosest(x: Int, y: Int, z: Int) =
        listOf(1,0,2)[abs(z - x).compareTo(abs(z - y))+1]

```
```rust

// 0ms
    pub fn find_closest(x: i32, y: i32, z: i32) -> i32 {
        [1,0,2][1+1.min((z-x).abs()-(z-y).abs()).max(-1) as usize]
    }

```
```c++

// 0ms
    int findClosest(int x, int y, int z) {
        return (abs(z-x)>abs(z-y))*2+(abs(z-x)<abs(z-y));
    }

```
```python

// 0ms
    findClosest=lambda _,x,y,z:(abs(z-x)>abs(z-y))<<1|(abs(z-x)<abs(z-y))

```

