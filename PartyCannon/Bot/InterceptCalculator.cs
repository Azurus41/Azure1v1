using System;
using Bot;
using RedUtils;
using RedUtils.Math;

public static class InterceptCalculator
{
	private const float MAX_SPEED = 2300f;

	private const float BOOST_ACCEL = 991.666f;

	private const float BOOST_CONSUMPTION = 33.333f;

	private const float THROTTLE_ACCEL = 1600f;

	private const float MAX_THROTTLE_SPEED = 1410f;

	private const float MAX_TURNING_SPEED = 1800f;

	private static readonly float[] BOOST_SPEEDS = new float[7] { 0f, 500f, 1000f, 1410f, 1800f, 2000f, 2300f };

	private static readonly float[] BOOST_TIMES = new float[7] { 0f, 0.5f, 1f, 1.4f, 1.8f, 2.1f, 2.4f };

	private static readonly float[] THROTTLE_SPEEDS = new float[4] { 0f, 500f, 1000f, 1410f };

	private static readonly float[] THROTTLE_TIMES = new float[4] { 0f, 0.8f, 1.6f, 2.5f };

	public static float EstimateTime(Car car, Vec3 target, bool allowBackwards = true)
	{
		float eta = EstimateTimeDirectional(car, target, 1);
		if (!allowBackwards)
		{
			return eta;
		}
		if (Math.Max(0f - car.Velocity.Dot(car.Forward), 0f) > 500f || car.Location.Dist(target) < 300f || car.Location.Dist(target) > 3000f)
		{
			float etaBackwards = EstimateTimeDirectional(car, target, -1) + 0.5f;
			return Math.Min(eta, etaBackwards);
		}
		return eta;
	}

	private static float EstimateTimeDirectional(Car car, Vec3 target, int direction)
	{
		float dist = car.Location.Dist(target);
		if (dist < 100f)
		{
			return 0f;
		}
		float speed = Math.Max(car.Velocity.Dot(car.Forward) * (float)direction, 0f);
		Vec3 dirToTarget = (target - car.Location).Normalize();
		float angleToTarget = (car.Forward * direction).Angle(dirToTarget);
		float turningRadius = GetTurningRadius(Math.Max(speed, 500f));
		float turnTime = angleToTarget * turningRadius / 1800f;
		if (angleToTarget < 0.5f)
		{
			turnTime = 0f;
		}
		float timeTotal = turnTime;
		float distLeft = dist - 200f;
		if (distLeft <= 0f)
		{
			return turnTime;
		}
		if (car.Boost > 0f && direction > 0)
		{
			float boostTime = Math.Min(car.Boost / 33.333f, 2.4f);
			float speedAfterBoost = InterpolateSpeed(BOOST_TIMES, BOOST_SPEEDS, boostTime, speed);
			float boostDist = (speed + speedAfterBoost) / 2f * boostTime;
			if (boostDist >= distLeft)
			{
				return timeTotal + SolveTimeForDistance(speed, 991.666f, distLeft);
			}
			timeTotal += boostTime;
			distLeft -= boostDist;
			speed = speedAfterBoost;
		}
		if (speed < 1410f && distLeft > 0f)
		{
			float timeToThrottleMax = (1410f - speed) / 1600f;
			float distThrottled = speed * timeToThrottleMax + 800f * timeToThrottleMax * timeToThrottleMax;
			if (distThrottled >= distLeft)
			{
				return timeTotal + SolveTimeForDistance(speed, 1600f, distLeft);
			}
			timeTotal += timeToThrottleMax;
			distLeft -= distThrottled;
			speed = 1410f;
		}
		if (distLeft > 0f)
		{
			timeTotal += distLeft / Math.Max(speed, 800f);
		}
		return timeTotal * 1.05f;
	}

	private static float GetTurningRadius(float speed)
	{
		float curvature = 0.0069f - 6.19E-06f * speed + 1.1E-08f * speed * speed;
		return 1f / Math.Max(curvature, 0.001f);
	}

	private static float InterpolateSpeed(float[] times, float[] speeds, float targetTime, float baseSpeed)
	{
		for (int i = 0; i < times.Length - 1; i++)
		{
			if (targetTime <= times[i + 1])
			{
				float frac = (targetTime - times[i]) / (times[i + 1] - times[i]);
				float speedInc = speeds[i] + frac * (speeds[i + 1] - speeds[i]);
				return baseSpeed + speedInc;
			}
		}
		return baseSpeed + speeds[^1];
	}

	private static float SolveTimeForDistance(float initialSpeed, float acceleration, float distance)
	{
		float time = initialSpeed * initialSpeed + 2f * acceleration * distance;
		if (time < 0f)
		{
			return distance / Math.Max(initialSpeed, 400f);
		}
		return (0f - initialSpeed + (float)Math.Sqrt(time)) / acceleration;
	}

	public static InterceptInfo CalculateInterceptAdvantage(Car[] ourTeam, Car[] opponents, Ball ball)
	{
		float ourBestTime = float.MaxValue;
		Car ourBestCar = null;
		Vec3 ourBestBallPosition = Ball.Location;
		float oppBestTime = float.MaxValue;
		Car opponentBestCar = null;
		Vec3 opponentBestBallPosition = Ball.Location;
		for (float t = 0f; t <= 3f; t += 0.1f)
		{
			Vec3 ballPos = Ball.Location + Ball.Velocity * t + 0.5f * new Vec3(0f, 0f, -650f) * t * t;
			ballPos.x = Math.Max(-4096f, Math.Min(4096f, ballPos.x));
			ballPos.y = Math.Max(-5120f, Math.Min(5120f, ballPos.y));
			ballPos.z = Math.Max(100f, ballPos.z);
			if (ourBestCar == null)
			{
				foreach (Car mate in ourTeam)
				{
					float eta = EstimateTime(mate, ballPos);
					if (eta <= t + 0.05f && eta < ourBestTime)
					{
						ourBestTime = eta;
						ourBestCar = mate;
						ourBestBallPosition = ballPos;
					}
				}
			}

			if (opponentBestCar == null)
			{
				foreach (Car opp in opponents)
				{
					float eta = EstimateTime(opp, ballPos);
					if (eta <= t + 0.05f && eta < oppBestTime)
					{
						oppBestTime = eta;
						opponentBestCar = opp;
						opponentBestBallPosition = ballPos;
					}
				}
			}

			if (ourBestCar != null && opponentBestCar != null)
			{
				break;
			}
		}
		
		return new InterceptInfo
		{
			OurBestTime = ourBestTime,
			OurBestCar = ourBestCar,
			OurBestBallPosition = ourBestBallPosition,
			OpponentBestTime = oppBestTime,
			OpponentBestCar = opponentBestCar,
			OpponentBestBallPosition = opponentBestBallPosition,
			OpponentsAreFaster = (oppBestTime < ourBestTime - 0.1f),
			WeAreFaster = (ourBestTime < oppBestTime - 0.1f)
		};
	}
}
