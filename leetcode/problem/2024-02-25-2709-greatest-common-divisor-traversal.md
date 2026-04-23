---
layout: leetcode-entry
title: "2709. Greatest Common Divisor Traversal"
permalink: "/leetcode/problem/2024-02-25-2709-greatest-common-divisor-traversal/"
leetcode_ui: true
entry_slug: "2024-02-25-2709-greatest-common-divisor-traversal"
---

[2709. Greatest Common Divisor Traversal](https://leetcode.com/problems/greatest-common-divisor-traversal/description/) hard
[blog post](https://leetcode.com/problems/greatest-common-divisor-traversal/solutions/4779877/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25022024-2709-greatest-common-divisor?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/peJkiq2EzDM)
![image.png](/assets/leetcode_daily_images/c6001b83.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/518

#### Problem TLDR

Are all numbers connected through gcd?

#### Intuition

The n^2 solution is trivial, just remember how to calculate the GCD.
Let's see how to optimize it by using all the possible hints and observing the example. To connect `4` to `3` we expect some number that are multiple of `2 and 3`. Those are prime numbers. It gives us the idea, that numbers can be connected throug the primes.

Let's build all the primes and assign our numbers to each. To build the primes, let's use Sieve of Eratosthenes https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes.

```bash
    // 4 3 12 8
    // 2 3 5 7 11 13 17 19 23 29 31
    // 4
    //   3
    //12 12
    // 8
```
In this example, we assign `4, 12 and 8` to prime `2`, `3 and 12` to prime 3. The two islands of primes `2` and `3` are connected through the number `12`.

Another example with the corner case of `1`:
![image.png](/assets/leetcode_daily_images/02b3ccc1.webp)

The different solution is to compute all the factors of each number and connect the numbers instead of the primes.

#### Approach

* use Union-Find and path compression `uf[x] = uf[uf[x]]`
* factors are less than `sqrt(n)`

#### Complexity
- Time complexity:
$$O(nsqrt(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin]

  fun canTraverseAllPairs(nums: IntArray): Boolean {
    if (nums.contains(1)) return nums.size == 1
    val nums = nums.toSet().toList()
    val p = BooleanArray(nums.max() + 1) { true }
    for (i in 2..sqrt(p.size.toDouble()).toInt()) if (p[i])
      for (j in i * i..<p.size step i) p[j] = false
    val primes = (2..<p.size).filter { p[it] }
    val uf = IntArray(primes.size) { it }
    fun Int.root(): Int {
      var x = this; while (x != uf[x]) x = uf[x]
      uf[this] = x; return x
    }
    val islands = HashSet<Int>()
    for (x in nums) {
      var prev = -1
      for (i in primes.indices) if (x % primes[i] == 0) {
          islands += i
          if (prev != -1) uf[prev.root()] = i.root()
          prev = i
        }
    }
    val oneOf = islands.firstOrNull()?.root() ?: -1
    return islands.all { it.root() == oneOf }
  }

```
```rust

    pub fn can_traverse_all_pairs(nums: Vec<i32>) -> bool {
      let mut uf: Vec<_> = (0..nums.len()).collect();
      fn root(uf: &mut Vec<usize>, mut x: usize) -> usize {
        while x != uf[x] { x = uf[x]; uf[x] = uf[uf[x]] } x}
      let mut mp = HashMap::<i32, usize>::new();
      for (i, &x) in nums.iter().enumerate() {
        if x == 1 { return nums.len() == 1 }
        let mut factors = vec![x];
        let mut a = x;
        for b in 2..=(x as f64).sqrt() as i32 {
          while a % b == 0 { a /= b; factors.push(b) }
        }
        if a > 1 { factors.push(a) }
        for &f in &factors {
          if let Some(&j) = mp.get(&f) {
            let ra = root(&mut uf, i);
            uf[ra] = root(&mut uf, j);
          }
          mp.insert(f, i);
        }
      }
      let ra = root(&mut uf, 0);
      (0..uf.len()).all(|b| root(&mut uf, b) == ra)
    }

```

