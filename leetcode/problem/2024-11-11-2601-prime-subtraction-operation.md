---
layout: leetcode-entry
title: "2601. Prime Subtraction Operation"
permalink: "/leetcode/problem/2024-11-11-2601-prime-subtraction-operation/"
leetcode_ui: true
entry_slug: "2024-11-11-2601-prime-subtraction-operation"
---

[2601. Prime Subtraction Operation](https://leetcode.com/problems/prime-subtraction-operation/description/) medium
[blog post](https://leetcode.com/problems/prime-subtraction-operation/solutions/6032862/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11112024-2601-prime-subtraction-operation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zflptogb9Lc)
[deep-dive](https://notebooklm.google.com/notebook/961ea1bd-c6c3-4439-8ce9-f821c746eec6/audio)
![1.webp](/assets/leetcode_daily_images/1840ac0e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/797

#### Problem TLDR

Increased sequence by subtracting primes? #medium #binary_search

#### Intuition

Go from back and decrease the number.
Example:

```j

    // 4 9 6 10
    //       *
    //     *
    //   *       9 -> 5 or less, diff >= 4, next prime after 4=5, 9-5=4
    //   4
    // *         4 -> 3 or less, diff >= 1, prime = 1
    // 3
    //
    // 2 2 ???? -> `1` is not a prime

```

#### Approach

* `1` is not a prime https://www.scientificamerican.com/blog/roots-of-unity/why-isnt-1-a-prime-number/
* prime numbers can be formed with Sieve of Eratosthenes: outer loop goes until `i = 2..sqrt(max)`, inner loop excludes all multipliers of `i, j += i`
* we can actually iterate forward in the array, subtract the largest prime
* we can Binary Search for prime

#### Complexity

- Time complexity:
$$O(n^2)$$ for the naive, $$O(sqrt(n) + nlog(n))$$ optimal

- Space complexity:
$$O(n)$$ for sieve, $$O(1)$$ if precomputed

#### Code

```kotlin

    fun primeSubOperation(nums: IntArray): Boolean {
        val primes = (2..<nums.max())
            .filter { i -> (2..<i).none { i % it == 0 } }
        return (nums.lastIndex - 1 downTo 0).all { i ->
            val diff = nums[i] - nums[i + 1] + 1
            if (diff > 0) nums[i] -= primes
                .firstOrNull { it >= diff } ?: return false
            nums[i] > 0
        }
    }

```
```rust

    pub fn prime_sub_operation(mut nums: Vec<i32>) -> bool {
        let primes: Vec<_> = (2..*nums.iter().max().unwrap())
            .filter(|&i| (2..i).all(|j| i % j > 0)).collect();
        (0..nums.len() - 1).rev().all(|i| {
            let diff = nums[i] - nums[i + 1] + 1;
            if diff > 0 {
                let p = primes.partition_point(|&x| x < diff);
                if p == primes.len() { return false }
                nums[i] -= primes[p];
            }
            nums[i] > 0
        })
    }

```
```c++

    bool primeSubOperation(vector<int>& nums) {
        vector<int> p(1001, 1); int prev = 0;
        for (int i = 2; i * i <= 1000; i++) if (p[i])
            for (int j = i * i; j <= 1000; j += i) p[j] = 0;
        for (int i = 0; i < nums.size(); ++i) {
            int diff = nums[i] - prev - 1;
            if (diff < 0) return 0;
            int j = diff; while (j > 1 && !p[j]) j--;
            if (j > 1) nums[i] -= j;
            prev = nums[i];
        }
        return 1;
    }

```

