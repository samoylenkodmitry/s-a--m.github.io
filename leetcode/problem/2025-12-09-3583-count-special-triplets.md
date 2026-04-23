---
layout: leetcode-entry
title: "3583. Count Special Triplets"
permalink: "/leetcode/problem/2025-12-09-3583-count-special-triplets/"
leetcode_ui: true
entry_slug: "2025-12-09-3583-count-special-triplets"
---

[3583. Count Special Triplets](https://leetcode.com/problems/count-special-triplets/description) medium
[blog post](https://leetcode.com/problems/count-special-triplets/solutions/7402202/kotlin-rust-by-samoylenkodmitry-1eh6/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09122025-3583-count-special-triplets?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pC5mSbjL0DY)

![e1399961-8558-4ab5-aa10-3dc1d63323b9 (1).webp](/assets/leetcode_daily_images/1536798f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1198

#### Problem TLDR

Count a[i]==a[j]*2==a[k] #medium #prefix_sum

#### Intuition

```j
    // 8 4 2 8 4
    //   *        look for all 8 left and right
    //
```

Count total frequency F.
Count frequency so-far L.
Add for each middle: L * (F - L)

#### Approach

* or a single pass solution from lee: count frequency so-far F, count pairs (2x,x) so-far C+=F[x], add each C[x/2]

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 176ms
    fun specialTriplets(n: IntArray) = {
        val f = LongArray(200002); val c = f.clone()
        n.sumOf { x -> val r = 1L*c[x/2]*(1 - x%2); c[x] += f[x*2]; f[x]++; r }
    }() % 1000000007
```
```rust
// 36ms
    pub fn special_triplets(n: Vec<i32>) -> i32 {
        let mut f = [0i32; 200002]; let mut c = f.clone();
        n.iter().fold(0, |r, &x|{ let x = x as usize;
            let rx = c[x>>1] * ((x&1)^1)as i32;
            c[x] = (c[x] + f[x<<1])%1000000007;
            f[x] += 1; (r + rx)%1000000007
        })
    }
```

