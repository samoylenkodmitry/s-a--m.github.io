---
layout: leetcode-entry
title: "735. Asteroid Collision"
permalink: "/leetcode/problem/2023-07-20-735-asteroid-collision/"
leetcode_ui: true
entry_slug: "2023-07-20-735-asteroid-collision"
---

[735. Asteroid Collision](https://leetcode.com/problems/asteroid-collision/description/) medium
[blog post](https://leetcode.com/problems/asteroid-collision/solutions/3790443/kotlin-stack/)
[substack](https://dmitriisamoilenko.substack.com/p/20072023-735-asteroid-collision?sd=pf)
![image.png](/assets/leetcode_daily_images/c2c3e174.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/281

#### Problem TLDR

Result after asteroids collide left-right exploding by size: `15 5 -15 -5 5 -> -15 -5 5`

#### Intuition

Let's add positive asteroids to the `Stack`. When negative met, it can fly over all smaller positive added, and can explode if larger met.

#### Approach

Kotlin's API helping reduce some LOC

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun asteroidCollision(asteroids: IntArray): IntArray = with(Stack<Int>()) {
        asteroids.forEach { sz ->
          if (!generateSequence { if (sz > 0 || isEmpty() || peek() < 0) null else peek() }
            .any {
              if (it <= -sz) pop()
              it >= -sz
            }) add(sz)
        }
        toIntArray()
    }

```

