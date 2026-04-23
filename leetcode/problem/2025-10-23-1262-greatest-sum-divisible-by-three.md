---
layout: leetcode-entry
title: "1262. Greatest Sum Divisible by Three"
permalink: "/leetcode/problem/2025-10-23-1262-greatest-sum-divisible-by-three/"
leetcode_ui: true
entry_slug: "2025-10-23-1262-greatest-sum-divisible-by-three"
---

[1262. Greatest Sum Divisible by Three](https://leetcode.com/problems/greatest-sum-divisible-by-three/description/) medium
[blog post](https://leetcode.com/problems/greatest-sum-divisible-by-three/solutions/7369140/kotlin-rust-by-samoylenkodmitry-fasc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23102025-1262-greatest-sum-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nayKlr2PK6c)

![d77898d7-d631-4c04-801f-267c931476bf (1).webp](/assets/leetcode_daily_images/e202e0a9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1182

#### Problem TLDR

Max sum that %3 #medium

#### Intuition

```j
    // add all numbers that are %3
    // search the remaining
    // they can be %3=1, %3=2
    // we need pairs 1+2
    // now take max pairs from 1+2
    // or tripples from %3=1 is this a choice?
    // big_1 big_1 big_1  small_2
    // this makes us loose big_1 + big_1 compared to small_2

    // what if we take from single sorted list
    // big_1 big_2 (take)
    // big_2 big_2 (skip) big_2(skip) big_1 big_1 wrong skip

    // what if we take from two lists big_1 sorted and big_2 sorted
    // take big_1 big_1 big_1 if it is bigger then big_1 big_2

    // edge case: 5 2 2 2  so three %3=2 is also %3

    // + some big corner case (my not optimal)
    // probably case where 111 == 222 == 12 and we have to choose
    // is this dp?

    // 23 minute, let's look hints: yes it is dp

    // 28 minute: MLE

    // i have another idea: all sum can be %3==0,1, or 2
    // if %3==0 just return sum
    // if %3==1 remove smallest %3==1 or two %3==2
    // if %3==2 remove smallest %3==2 or two %3==1
```

* remove the smallest from sum
* only 3 cases possible

#### Approach

* another approach is dp: keep three running sums: %3=0,1,2; evaluate where to place

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 4ms
    fun maxSumDivThree(n: IntArray): Int {
        var s=0; var s1=20000; var s11=s1; var s2=s1; var s22=s1
        for (x in n) {
            s += x
            if (x%3==1) if (x<s1) {s11=s1; s1=x} else if (x<s11) s11=x
            if (x%3==2) if (x<s2) {s22=s2; s2=x} else if (x<s22) s22=x
        }
        return s - if (s%3==2) min(s2,s1+s11) else (s%3)*min(s1,s2+s22)
    }
```
```rust
// 0ms
    pub fn max_sum_div_three(n: Vec<i32>) -> i32 {
        let mut s = [0,0,0];
        for x in n { for y in [x+s[0], x+s[1], x+s[2]] {
            let i = (y%3) as usize; s[i] = s[i].max(y) }}; s[0]
    }
```

