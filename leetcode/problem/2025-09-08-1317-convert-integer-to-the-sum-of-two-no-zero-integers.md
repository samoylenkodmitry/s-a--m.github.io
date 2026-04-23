---
layout: leetcode-entry
title: "1317. Convert Integer to the Sum of Two No-Zero Integers"
permalink: "/leetcode/problem/2025-09-08-1317-convert-integer-to-the-sum-of-two-no-zero-integers/"
leetcode_ui: true
entry_slug: "2025-09-08-1317-convert-integer-to-the-sum-of-two-no-zero-integers"
---

[1317. Convert Integer to the Sum of Two No-Zero Integers](https://leetcode.com/problems/convert-integer-to-the-sum-of-two-no-zero-integers/description/) easy
[blog post](https://leetcode.com/problems/convert-integer-to-the-sum-of-two-no-zero-integers/solutions/7168318/kotlin-rust-by-samoylenkodmitry-lwut/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08092025-1317-convert-integer-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/256AzPk4hvU)

![1.webp](/assets/leetcode_daily_images/9bd816a2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1106

#### Problem TLDR

Find a+b=n without zeros in digits #easy

#### Intuition

Brute force `a=1..<n, b = n-a`

#### Approach

* any faster solution?
* any shorter solution?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 55ms
    fun getNoZeroIntegers(n: Int) =
    (0..n).map { listOf(it, n-it) }.find { '0' !in "$it" }

```
```rust

// 0ms
    pub fn get_no_zero_integers(n: i32) -> Vec<i32> {
       (0..n).map(|a|vec![a,n-a]).find(|v|!format!("{:?}",v).contains('0')).unwrap()
    }

```
```c++

// 0ms
    vector<int> getNoZeroIntegers(int n) {
        int a=0,b=0,s=1;
        while (n) {
            int d = n%10; n/=10;
            if (n&&d<2) a += s*(8+d), b += s<<1, --n;
            else a += s, b += s*(d-1);
            s *= 10;
        }
        return {a,b};
    }

```

